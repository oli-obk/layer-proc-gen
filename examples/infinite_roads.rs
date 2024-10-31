use ::rand::distributions::uniform::SampleRange as _;
use arrayvec::ArrayVec;
use generic_layers::{rng_for_point, UniformPointLayer};
use macroquad::prelude::*;
use miniquad::window::screen_size;
use std::{
    borrow::Borrow,
    cell::{Cell, Ref, RefCell},
    f32::consts::PI,
    num::NonZeroU8,
    sync::Arc,
};

use layer_proc_gen::*;
use rigid2d::Body;
use rolling_grid::{GridIndex, GridPoint};
use vec2::{Bounds, Line, Num, Point2d};

#[path = "../tests/tracing.rs"]
mod tracing_helper;
use tracing_helper::*;

#[derive(PartialEq, Debug, Clone, Default)]
struct City {
    center: Point2d,
    size: i64,
    name: String,
}

impl From<Point2d> for City {
    fn from(center: Point2d) -> Self {
        let mut rng = rng_for_point::<0, _>(center);
        City {
            center,
            size: { (100..500).sample_single(&mut rng) },
            name: (0..(3..12).sample_single(&mut rng))
                .map(|_| ('a'..='z').sample_single(&mut rng))
                .collect(),
        }
    }
}

/// Removes locations that are too close to others
struct ReducedCities {
    cities: LayerDependency<UniformPointLayer<City, 11, 1>>,
}

#[derive(PartialEq, Debug, Clone, Default)]
struct ReducedCitiesChunk {
    points: ArrayVec<City, 7>,
}

impl Layer for ReducedCities {
    type Chunk = ReducedCitiesChunk;
    type Store<T> = Arc<T>;

    fn ensure_all_deps(&self, chunk_bounds: Bounds) {
        self.cities.ensure_loaded_in_bounds(chunk_bounds);
    }
}

impl Chunk for ReducedCitiesChunk {
    type Layer = ReducedCities;
    type Store = Self;
    const SIZE: Point2d<u8> = Point2d::splat(11);

    fn compute(layer: &Self::Layer, index: GridPoint<Self>) -> Self {
        let mut points = ArrayVec::new();
        'points: for p in layer
            .cities
            .get_or_compute(index.into_same_chunk_size())
            .points
        {
            for other in layer.cities.get_range(Bounds {
                min: p.center,
                max: p.center + Point2d::splat(p.size),
            }) {
                for other in other.points {
                    if other == p {
                        continue;
                    }

                    if other.center.manhattan_dist(p.center) < p.size + other.size
                        && p.size < other.size
                    {
                        continue 'points;
                    }
                }
            }
            points.push(p);
        }
        ReducedCitiesChunk { points }
    }
}

/// Removes locations that are too close to others
struct ReducedLocations {
    raw_locations: LayerDependency<UniformPointLayer<Point2d, 6, 0>>,
    cities: LayerDependency<ReducedCities>,
}

#[derive(PartialEq, Debug, Clone, Default)]
struct ReducedLocationsChunk {
    points: ArrayVec<Point2d, 7>,
}

impl Layer for ReducedLocations {
    type Chunk = ReducedLocationsChunk;
    type Store<T> = Arc<T>;

    fn ensure_all_deps(&self, chunk_bounds: Bounds) {
        self.raw_locations.ensure_loaded_in_bounds(chunk_bounds);
        self.cities.ensure_loaded_in_bounds(chunk_bounds);
    }
}

impl Chunk for ReducedLocationsChunk {
    type Layer = ReducedLocations;
    type Store = Self;
    const SIZE: Point2d<u8> = Point2d::splat(6);

    fn compute(layer: &Self::Layer, index: GridPoint<Self>) -> Self {
        let bounds = Self::bounds(index);
        let center = bounds.center();
        if layer.cities.get_range(bounds).all(|cities| {
            cities
                .points
                .iter()
                .all(|city| center.manhattan_dist(city.center) > city.size)
        }) {
            return Self::default();
        }
        let mut points = ArrayVec::new();
        'points: for p in layer
            .raw_locations
            .get_or_compute(index.into_same_chunk_size())
            .points
        {
            for other in layer.raw_locations.get_range(Bounds {
                min: p,
                max: p + Point2d::splat(15),
            }) {
                for other in other.points {
                    if other == p {
                        continue;
                    }
                    if other.dist_squared(p) < 15 * 15 {
                        continue 'points;
                    }
                }
            }
            points.push(p);
        }
        ReducedLocationsChunk { points }
    }
}

struct Roads {
    locations: LayerDependency<ReducedLocations>,
}

#[derive(PartialEq, Debug, Default)]
struct RoadsChunk {
    roads: Vec<Line>,
}

impl Layer for Roads {
    type Chunk = RoadsChunk;
    type Store<T> = T;

    #[track_caller]
    fn ensure_all_deps(&self, chunk_bounds: Bounds) {
        self.locations.ensure_loaded_in_bounds(chunk_bounds);
    }
}

impl Chunk for RoadsChunk {
    type Layer = Roads;
    type Store = Arc<Self>;
    const SIZE: Point2d<u8> = Point2d::splat(6);

    fn compute(layer: &Self::Layer, index: GridPoint<Self>) -> Self::Store {
        let roads = gen_roads(
            layer
                .locations
                .get_grid_range(
                    Bounds::point(index.into_same_chunk_size()).pad(Point2d::splat(GridIndex::ONE)),
                )
                .map(|chunk| chunk.points),
            |&p| p,
            |&a, &b| a.to(b),
        );
        RoadsChunk { roads }.into()
    }
}

fn gen_roads<T: Clone, U>(
    chunks: impl Iterator<Item = impl Borrow<[T]>>,
    get_point: impl Fn(&T) -> Point2d,
    mk: impl Fn(&T, &T) -> U,
) -> Vec<U> {
    let mut roads = vec![];
    let mut points: ArrayVec<T, { 3 * 9 }> = ArrayVec::new();
    let mut start = usize::MAX;
    let mut n = usize::MAX;
    for (i, grid) in chunks.enumerate() {
        let grid = grid.borrow();
        if i == 4 {
            start = points.len();
            n = grid.len();
        }
        points.extend(grid.iter().cloned());
    }
    // We only care about the roads starting from the center grid cell, as the others are not necessarily correct,
    // or will be computed by the other grid cells.
    // The others may connect the outer edges of the current grid range and thus connect roads that
    // don't satisfy the algorithm.
    // This algorithm is https://en.m.wikipedia.org/wiki/Relative_neighborhood_graph adjusted for
    // grid-based computation. It's a brute force implementation, but I think that is faster than going through
    // a Delaunay triangulation first, as instead of (3*9)^3 = 19683 inner loop iterations we have only
    // 3 * (2 + 1 + 3*4) * 3*9 = 1215
    // FIXME: cache distance computations as we do them, we can save 1215-(3*9^3)/2 = 850 distance computations (70%) and figure
    // out how to cache them across grid cells (along with removing them from the cache when they aren't needed anymore)
    // as the neighboring cells will be redoing the same distance computations.
    for (i, a_val) in points.iter().enumerate().skip(start).take(n) {
        let a = get_point(a_val);
        for b_val in points.iter().skip(i + 1) {
            let b = get_point(b_val);
            let dist = a.dist_squared(b);
            if points.iter().all(|c| {
                let c = get_point(c);
                if a == c || b == c {
                    return true;
                }
                // FIXME: make cheaper by already bailing if `x*x` is larger than dist,
                // to avoid computing `y*y`.
                let a_dist = a.dist_squared(c);
                let b_dist = c.dist_squared(b);
                dist < a_dist || dist < b_dist
            }) {
                roads.push(mk(a_val, b_val))
            }
        }
    }
    roads
}

struct Highways {
    cities: LayerDependency<ReducedCities>,
    locations: LayerDependency<ReducedLocations>,
}

#[derive(PartialEq, Debug, Clone)]

struct Highway {
    line: Line,
    start_city: String,
    start_sign: String,
    end_city: String,
    end_sign: String,
}

#[derive(PartialEq, Debug, Default)]
struct HighwaysChunk {
    roads: Vec<Highway>,
}

impl Layer for Highways {
    type Chunk = HighwaysChunk;
    type Store<T> = T;

    #[track_caller]
    fn ensure_all_deps(&self, chunk_bounds: Bounds) {
        self.cities.ensure_loaded_in_bounds(chunk_bounds);
    }
}

impl Chunk for HighwaysChunk {
    type Layer = Highways;
    type Store = Arc<Self>;
    const SIZE: Point2d<u8> = ReducedCitiesChunk::SIZE;

    fn compute(layer: &Self::Layer, index: GridPoint<Self>) -> Self::Store {
        let roads = gen_roads(
            layer
                .cities
                .get_grid_range(
                    Bounds::point(index.into_same_chunk_size()).pad(Point2d::splat(GridIndex::ONE)),
                )
                .map(|chunk| chunk.points),
            |p| p.center,
            |a, b| {
                (
                    a.size,
                    b.size,
                    a.center.to(b.center),
                    a.name.clone(),
                    b.name.clone(),
                )
            },
        );

        let roads = roads
            .into_iter()
            .map(|(start_size, end_size, road, start_city, end_city)| {
                let approx_start = road.with_manhattan_length(start_size).end;
                let approx_end = road.flip().with_manhattan_length(end_size).end;

                let closest = |p, start| {
                    let mut closest = None;
                    Chunk::pos_to_grid(p)
                        .to(Chunk::pos_to_grid(start))
                        .iter_all_touched_pixels(|x, y| {
                            let index = Point2d::new(x, y);
                            closest = layer
                                .locations
                                .get_or_compute(index)
                                .points
                                .iter()
                                .copied()
                                .chain(closest)
                                .min_by_key(|point| point.dist_squared(p))
                        });
                    closest
                };
                let line = Line {
                    start: closest(approx_start, road.start).unwrap_or(approx_start),
                    end: closest(approx_end, road.end).unwrap_or(approx_end),
                };
                let dist_km = ((line.len_squared() as f32).sqrt() / 1000.).ceil();
                Highway {
                    line,
                    start_sign: format!("{end_city} {dist_km}km"),
                    end_sign: format!("{start_city} {dist_km}km"),
                    start_city,
                    end_city,
                }
            })
            .collect();
        HighwaysChunk { roads }.into()
    }
}

struct Player {
    roads: LayerDependency<Roads>,
    highways: LayerDependency<Highways>,
    max_zoom_in: NonZeroU8,
    max_zoom_out: NonZeroU8,
    car: Car,
    last_grid_vision_range: Cell<(
        Bounds<GridIndex<RoadsChunk>>,
        Bounds<GridIndex<HighwaysChunk>>,
    )>,
    roads_for_last_grid_vision_range: RefCell<Vec<Highway>>,
}

impl Player {
    pub fn new(roads: Roads, highways: Highways) -> Self {
        Self {
            roads: roads.into_dep(),
            highways: highways.into_dep(),
            max_zoom_in: NonZeroU8::new(5).unwrap(),
            max_zoom_out: NonZeroU8::new(10).unwrap(),
            car: Car {
                length: 4.,
                width: 2.,
                body: Body {
                    position: vec2(-56., -43.),
                    rotation: -0.75,
                    ..Default::default()
                },
                steering_limit: 15,
                steering: 0.,
                color: DARKPURPLE,
                braking: false,
                reversing: false,
            },
            last_grid_vision_range: (
                Bounds::point(Point2d::splat(GridIndex::from_raw(0))),
                Bounds::point(Point2d::splat(GridIndex::from_raw(0))).into(),
            )
                .into(),
            roads_for_last_grid_vision_range: vec![].into(),
        }
    }

    /// Absolute position and function to go from a global position
    /// to one relative to the player.
    pub fn point2screen(&self) -> impl Fn(Point2d) -> Vec2 {
        let player_pos = self.pos();

        // Avoid moving everything in whole pixels and allow for smooth sub-pixel movement instead
        let adjust = self.car.body.position.fract();
        move |point: Point2d| -> Vec2 {
            let point = point - player_pos;
            i64vec2(point.x, point.y).as_vec2() - adjust
        }
    }

    fn pos(&self) -> Point2d {
        let player_pos = Point2d {
            x: self.car.body.position.x as i64,
            y: self.car.body.position.y as i64,
        };
        player_pos
    }

    pub fn vision_range<C: Chunk>(&self, half_screen_visible_area: Vec2) -> Bounds {
        let padding = half_screen_visible_area.abs().ceil().as_i64vec2();
        let padding = Point2d::new(padding.x as i64, padding.y as i64);
        let mut vision_range = Bounds::point(self.pos()).pad(padding);
        let padding = C::SIZE.map(|i| 1 << i);
        vision_range.min -= padding;
        vision_range.max += padding;
        vision_range
    }

    pub fn grid_vision_range<C: Chunk>(
        &self,
        half_screen_visible_area: Vec2,
    ) -> Bounds<GridIndex<C>> {
        C::bounds_to_grid(self.vision_range::<C>(half_screen_visible_area))
    }

    pub fn roads(&self, half_screen_visible_area: Vec2) -> Ref<'_, [Highway]> {
        let grid_vision_range = self.grid_vision_range(half_screen_visible_area);
        let highway_vision_range = self.grid_vision_range(half_screen_visible_area);
        if (grid_vision_range, highway_vision_range) != self.last_grid_vision_range.get() {
            self.last_grid_vision_range
                .set((grid_vision_range, highway_vision_range));
            let mut roads = self.roads_for_last_grid_vision_range.borrow_mut();
            roads.clear();
            for index in grid_vision_range.iter() {
                for &line in &self.roads.get_or_compute(index).roads {
                    roads.push(Highway {
                        line,
                        start_city: String::new(),
                        start_sign: String::new(),
                        end_city: String::new(),
                        end_sign: String::new(),
                    });
                }
            }
            for index in highway_vision_range.iter() {
                roads.extend_from_slice(&self.highways.get_or_compute(index).roads);
            }
        }
        Ref::map(self.roads_for_last_grid_vision_range.borrow(), |r| &**r)
    }
}

#[macroquad::main("layer proc gen demo")]
async fn main() {
    init_tracing();

    let mut camera = Camera2D::default();
    let standard_zoom = Vec2::from(screen_size()).recip() * 4.;
    camera.zoom = standard_zoom;
    set_camera(&camera);
    let mut overlay_camera = Camera2D::default();
    overlay_camera.zoom = standard_zoom / 4.;
    overlay_camera.offset = vec2(-1., 1.);

    let cities = UniformPointLayer::new();
    let cities = ReducedCities { cities }.into_dep();
    let locations = ReducedLocations {
        raw_locations: Layer::new(),
        cities: cities.clone(),
    }
    .into_dep();
    let roads = Roads {
        locations: locations.clone(),
    };
    let highways = Highways {
        cities: cities.clone(),
        locations,
    };
    let mut player = Player::new(roads, highways);
    let mut smooth_cam_speed = 0.0;
    let mut debug_zoom = 1.0;

    loop {
        player.car.update(Actions {
            accelerate: is_key_down(KeyCode::W),
            reverse: is_key_down(KeyCode::S),
            hand_brake: is_key_down(KeyCode::Space),
            left: is_key_down(KeyCode::A),
            right: is_key_down(KeyCode::D),
        });
        if is_key_pressed(KeyCode::Up) {
            debug_zoom *= 2.0;
        }
        if is_key_pressed(KeyCode::Down) {
            debug_zoom /= 2.0;
        }

        smooth_cam_speed = smooth_cam_speed * 0.99 + player.car.body.velocity.length() / 30. * 0.01;
        let max_zoom_in = f32::from(player.max_zoom_in.get());
        let max_zoom_out = f32::from(player.max_zoom_out.get());
        smooth_cam_speed = smooth_cam_speed.clamp(0.0, max_zoom_in);
        camera.zoom = standard_zoom * (max_zoom_in + 1.0 / max_zoom_out - smooth_cam_speed);
        camera.zoom /= debug_zoom;
        set_camera(&camera);
        camera.zoom *= debug_zoom;

        let point2screen = player.point2screen();
        clear_background(DARKGREEN);

        let draw_bounds = |bounds: Bounds| {
            if debug_zoom == 1.0 {
                return;
            }
            let min = point2screen(bounds.min);
            let max = point2screen(bounds.max);
            draw_rectangle_lines(
                min.x as f32,
                min.y as f32,
                (max.x - min.x) as f32,
                (max.y - min.y) as f32,
                debug_zoom,
                PURPLE,
            );
        };

        let padding = camera.screen_to_world(Vec2::splat(0.));
        if debug_zoom != 1. {
            draw_rectangle_lines(
                -padding.x,
                -padding.y,
                padding.x * 2.,
                padding.y * 2.,
                debug_zoom,
                PURPLE,
            );
        }
        let vision_range = player.vision_range::<RoadsChunk>(padding);
        draw_bounds(vision_range);

        for index in player.grid_vision_range(padding).iter() {
            let current_chunk = RoadsChunk::bounds(index);
            draw_bounds(current_chunk);
        }

        let draw_line = |line: Line, thickness, color| {
            let start = point2screen(line.start);
            let end = point2screen(line.end);
            draw_line(start.x, start.y, end.x, end.y, thickness, color);
        };

        for highway in player.roads(padding).iter() {
            let start = point2screen(highway.line.start);
            let end = point2screen(highway.line.end);
            draw_line(highway.line, 8., GRAY);
            draw_circle(start.x, start.y, 4., GRAY);
            draw_circle(start.x, start.y, 0.1, WHITE);
            draw_circle(end.x, end.y, 4., GRAY);
            draw_circle(end.x, end.y, 0.1, WHITE);
            for (start, end, sign, name) in [
                (start, end, &highway.start_sign, &highway.start_city),
                (end, start, &highway.end_sign, &highway.end_city),
            ] {
                if sign.is_empty() && name.is_empty() {
                    continue;
                }
                let direction = end - start;
                let mut rotation = direction.to_angle();
                let mut sign_offset = 6. * 0.2;
                let mut name_offset = 14. * 0.2;
                let mut sign_line_distance = -1.;
                if rotation.abs() < PI / 2. {
                    std::mem::swap(&mut sign_offset, &mut name_offset);
                    sign_line_distance *= -1.;
                }
                if rotation > PI / 2. {
                    rotation -= PI;
                } else if rotation < -PI / 2. {
                    rotation += PI;
                }
                let pos = start
                    + direction.perp().normalize() * (sign_offset + 4.)
                    + direction.normalize() * 100.;
                draw_multiline_text_ex(
                    sign,
                    pos.x,
                    pos.y,
                    // bug in macroquad: line distance factor is applied on y axis, ignoring rotation
                    Some(sign_line_distance),
                    TextParams {
                        font_size: 20,
                        font_scale: 0.2,
                        rotation,
                        color: WHITE,
                        ..Default::default()
                    },
                );
                let pos = start - direction.perp().normalize() * name_offset
                    + direction.normalize() * 50.;
                draw_text_ex(
                    name,
                    pos.x,
                    pos.y,
                    TextParams {
                        font_size: 20,
                        font_scale: 0.2,
                        rotation,
                        color: WHITE,
                        ..Default::default()
                    },
                );
            }
        }
        for highway in player.roads(padding).iter() {
            draw_line(highway.line, 0.2, WHITE);
        }

        if debug_zoom != 1.0 {
            for &road in player
                .roads
                .get_or_compute(RoadsChunk::pos_to_grid(player.pos()))
                .roads
                .iter()
            {
                draw_line(road, debug_zoom, PURPLE)
            }
            for city in cities
                .get_or_compute(ReducedCitiesChunk::pos_to_grid(player.pos()))
                .points
                .iter()
            {
                let pos = point2screen(city.center);
                draw_circle_lines(pos.x, pos.y, city.size as f32, debug_zoom, PURPLE);
            }
        }

        player.car.draw();

        if debug_zoom != 1.0 {
            set_camera(&overlay_camera);
            draw_text(&format!("fps: {}", get_fps()), 0., 30., 30., WHITE);
            draw_text(
                &format!(
                    "speed: {:.0}km/h",
                    player.car.body.velocity.length() * 3600. / 1000.
                ),
                0.,
                60.,
                30.,
                WHITE,
            );
            draw_multiline_text(
                &format!("{:#.2?}", player.car.body),
                0.,
                90.,
                30.,
                Some(1.),
                WHITE,
            );
        }

        next_frame().await
    }
}

#[derive(Debug)]
struct Car {
    // In `m`
    length: f32,
    // In `m`
    width: f32,
    body: Body,
    color: Color,
    /// Maximum angle of the front wheels, in degrees
    steering_limit: i8,
    steering: f32,
    /// Enable the braking lights
    braking: bool,
    /// Enable the reversing lights
    reversing: bool,
}

struct Actions {
    accelerate: bool,
    hand_brake: bool,
    reverse: bool,
    left: bool,
    right: bool,
}

const ENGINE_POWER: f32 = 5.;
const FRICTION: f32 = -0.0005;
const DRAG: f32 = -0.005;
const MAX_WHEEL_FRICTION_BEFORE_SLIP: f32 = 20.;

impl Car {
    // Taken from https://github.com/godotrecipes/2d_car_steering/blob/master/car.gd
    fn update(&mut self, actions: Actions) {
        let heading = Vec2::from_angle(self.body.rotation);
        // Get Inputs

        const STEERING_SPEED: f32 = 2.;
        self.steering += if actions.left {
            -STEERING_SPEED
        } else if actions.right {
            STEERING_SPEED
        } else {
            if self.steering.abs() < STEERING_SPEED {
                -self.steering
            } else {
                -self.steering.signum() * STEERING_SPEED
            }
        };
        self.steering = self
            .steering
            .clamp((-self.steering_limit).into(), self.steering_limit.into());
        let steer_dir = f32::from(self.steering).to_radians();

        self.braking = actions.hand_brake;
        self.reversing = actions.reverse;

        // Kill all movement once the car gets slow enough
        // (instead of getting closer to zero velocity in decreasingly small steps)
        if !actions.accelerate && !actions.reverse && self.body.velocity.length() < 0.05 {
            self.body.velocity = Vec2::ZERO
        }

        // Apply drag (from air on car), effectively specifying a maximum velocity.
        self.body.add_impulse(
            Vec2::ZERO,
            self.body.velocity * self.body.velocity.length() * DRAG,
        );

        let wheel_offset = heading * self.length / 2.0;

        // Calculate wheel friction forces
        let rear_impulse = self.wheel_velocity(heading, -wheel_offset, actions.hand_brake);
        let rear_impulse = slip(rear_impulse);
        self.body.add_impulse(-wheel_offset, rear_impulse);

        let front_wheel_direction = Vec2::from_angle(steer_dir).rotate(heading);
        let front_impulse = self.wheel_velocity(front_wheel_direction, wheel_offset, false);
        let front_impulse = slip(front_impulse);
        self.body.add_impulse(wheel_offset, front_impulse);

        // Accumulate car engine and brake behaviors
        if actions.reverse {
            self.body
                .add_impulse(wheel_offset, -front_wheel_direction * ENGINE_POWER);
        } else if actions.accelerate {
            let multiplier = if is_key_down(KeyCode::LeftShift) {
                10.
            } else {
                1.
            };
            self.body.add_impulse(
                wheel_offset,
                front_wheel_direction * ENGINE_POWER * multiplier,
            );
        }

        self.body.step(get_frame_time());
    }

    /// Compute and aggregate lateral and forward friction.
    /// friction is infinite up to a limit where the wheel slips (for drifting)
    fn wheel_velocity(&mut self, direction: Vec2, wheel_position: Vec2, braking: bool) -> Vec2 {
        let normal = direction.perp();
        let velocity = self.body.velocity_at_local_point(wheel_position);
        let lateral_velocity = velocity.dot(normal) * normal;
        let forward_velocity = velocity.dot(direction) * direction;
        if braking {
            -forward_velocity - lateral_velocity
        } else {
            forward_velocity * FRICTION - lateral_velocity
        }
    }

    fn draw(&self) {
        draw_rectangle_ex(
            0.,
            0.,
            self.length,
            self.width,
            DrawRectangleParams {
                offset: vec2(0.5, 0.5),
                rotation: self.body.rotation,
                color: self.color,
            },
        );
        let rotation = Vec2::from_angle(self.body.rotation) * self.length / 2.;
        draw_circle(rotation.x, rotation.y, self.width / 2., self.color);

        if self.braking || self.reversing {
            let rotation = Vec2::from_angle(self.body.rotation) * (self.length / 2. + 1.);
            draw_rectangle_ex(
                -rotation.x,
                -rotation.y,
                2.,
                self.width,
                DrawRectangleParams {
                    offset: vec2(0.5, 0.5),
                    rotation: self.body.rotation,
                    color: if self.reversing { WHITE } else { RED },
                },
            );
        }

        draw_rectangle_ex(
            rotation.x,
            rotation.y,
            2.,
            1.,
            DrawRectangleParams {
                offset: vec2(0.5, 0.5),
                rotation: self.steering.to_radians() + self.body.rotation,
                color: BLACK,
            },
        );
        draw_rectangle_ex(
            -rotation.x,
            -rotation.y,
            2.,
            1.,
            DrawRectangleParams {
                offset: vec2(0., 0.5),
                rotation: self.body.rotation,
                color: BLACK,
            },
        );

        draw_line(
            0.,
            0.,
            self.body.velocity.x,
            self.body.velocity.y,
            1.,
            YELLOW,
        );
    }
}

// Reduce force if aboove the slip limit
fn slip(friction: Vec2) -> Vec2 {
    friction.clamp_length_max(MAX_WHEEL_FRICTION_BEFORE_SLIP)
}
