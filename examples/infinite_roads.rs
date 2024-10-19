use ::rand::prelude::*;
use ::tracing::{debug, trace};
use arrayvec::ArrayVec;
use macroquad::{prelude::*, time};
use miniquad::window::screen_size;
use std::{sync::Arc, vec};

use layer_proc_gen::*;
use rolling_grid::{GridIndex, GridPoint, RollingGrid};
use vec2::{Bounds, Line, Point2d};

#[path = "../tests/tracing.rs"]
mod tracing_helper;
use tracing_helper::*;

#[derive(Default)]
struct Locations(RollingGrid<Self>);
#[derive(PartialEq, Debug, Clone)]
struct LocationsChunk {
    points: [Point2d; 3],
}

impl Layer for Locations {
    type Chunk = LocationsChunk;

    fn rolling_grid(&self) -> &RollingGrid<Self> {
        &self.0
    }

    fn ensure_all_deps(&self, _chunk_bounds: Bounds) {}
}

impl Chunk for LocationsChunk {
    type Layer = Locations;
    type Store = Self;

    fn compute(_layer: &Self::Layer, index: GridPoint) -> Self {
        let chunk_bounds = Self::bounds(index);
        trace!(?chunk_bounds);
        let mut x = SmallRng::seed_from_u64(index.x.0 as u64);
        let mut y = SmallRng::seed_from_u64(index.y.0 as u64);
        let mut seed = [0; 32];
        x.fill_bytes(&mut seed[..16]);
        y.fill_bytes(&mut seed[16..]);
        let mut rng = SmallRng::from_seed(seed);
        let points = [
            chunk_bounds.sample(&mut rng),
            chunk_bounds.sample(&mut rng),
            chunk_bounds.sample(&mut rng),
        ];
        debug!(?points);
        LocationsChunk { points }
    }
}

/// Removes locations that are too close to others
struct ReducedLocations {
    grid: RollingGrid<Self>,
    raw_locations: LayerDependency<Locations, 0, 0>,
}

#[derive(PartialEq, Debug, Clone)]
struct ReducedLocationsChunk {
    points: ArrayVec<Point2d, 3>,
}

impl Layer for ReducedLocations {
    type Chunk = ReducedLocationsChunk;

    fn rolling_grid(&self) -> &RollingGrid<Self> {
        &self.grid
    }

    fn ensure_all_deps(&self, chunk_bounds: Bounds) {
        self.raw_locations.ensure_loaded_in_bounds(chunk_bounds);
    }
}

impl Chunk for ReducedLocationsChunk {
    type Layer = ReducedLocations;
    type Store = Self;

    fn compute(layer: &Self::Layer, index: GridPoint) -> Self {
        let mut points = ArrayVec::new();
        'points: for p in layer.raw_locations.get_or_compute(index).points {
            for other in layer.raw_locations.get_range(Bounds {
                min: p,
                max: p + Point2d::splat(100),
            }) {
                for other in other.points {
                    if other == p {
                        continue;
                    }
                    if other.dist_squared(p) < 100 * 100 {
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
    grid: RollingGrid<Self>,
    locations: LayerDependency<ReducedLocations, 256, 256>,
}

#[derive(PartialEq, Debug)]
struct RoadsChunk {
    roads: Vec<Line>,
}

impl Layer for Roads {
    type Chunk = RoadsChunk;

    fn rolling_grid(&self) -> &RollingGrid<Self> {
        &self.grid
    }

    #[track_caller]
    fn ensure_all_deps(&self, chunk_bounds: Bounds) {
        self.locations.ensure_loaded_in_bounds(chunk_bounds);
    }
}

impl Chunk for RoadsChunk {
    type Layer = Roads;
    type Store = Arc<Self>;

    fn compute(layer: &Self::Layer, index: GridPoint) -> Self::Store {
        let mut roads = vec![];
        let mut points: ArrayVec<Point2d, { 3 * 9 }> = ArrayVec::new();
        let mut start = usize::MAX;
        let mut n = usize::MAX;
        for (i, grid) in layer
            .locations
            .get_grid_range(Bounds::point(index).pad(Point2d::splat(1).map(GridIndex)))
            .enumerate()
        {
            if i == 4 {
                start = points.len();
                n = grid.points.len();
            }
            points.extend(grid.points.iter().copied());
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
        for (i, &a) in points.iter().enumerate().skip(start).take(n) {
            for &b in points.iter().skip(i + 1) {
                let dist = a.dist_squared(b);
                if points.iter().copied().all(|c| {
                    if a == c || b == c {
                        return true;
                    }
                    // FIXME: make cheaper by already bailing if `x*x` is larger than dist,
                    // to avoid computing `y*y`.
                    let a_dist = a.dist_squared(c);
                    let b_dist = c.dist_squared(b);
                    dist < a_dist || dist < b_dist
                }) {
                    roads.push(a.to(b))
                }
            }
        }
        debug!(?roads);
        RoadsChunk { roads }.into()
    }
}

struct Player {
    grid: RollingGrid<Self>,
    roads: LayerDependency<Roads, 1000, 1000>,
}

impl Player {
    pub fn new(roads: Arc<Roads>) -> Self {
        Self {
            grid: Default::default(),
            roads: roads.into(),
        }
    }
}

#[derive(PartialEq, Debug, Clone)]
struct PlayerChunk;

impl Layer for Player {
    type Chunk = PlayerChunk;

    fn rolling_grid(&self) -> &RollingGrid<Self> {
        &self.grid
    }

    const GRID_SIZE: Point2d<u8> = Point2d::splat(3);

    const GRID_OVERLAP: u8 = 2;

    fn ensure_all_deps(&self, chunk_bounds: Bounds) {
        self.roads.ensure_loaded_in_bounds(chunk_bounds);
    }
}

impl Chunk for PlayerChunk {
    type Layer = Player;
    type Store = Self;

    fn compute(_layer: &Self::Layer, _index: GridPoint) -> Self {
        PlayerChunk
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
    overlay_camera.zoom = standard_zoom;
    overlay_camera.offset = vec2(-1., 1.);

    let raw_locations = Arc::new(Locations::default());
    let locations = Arc::new(ReducedLocations {
        grid: Default::default(),
        raw_locations: raw_locations.into(),
    });
    let roads = Arc::new(Roads {
        grid: Default::default(),
        locations: locations.into(),
    });
    let player = Player::new(roads.clone());
    let mut last_load_time = 0.;
    let mut smooth_cam_rotation = 0.0;
    let mut smooth_cam_speed = 0.0;
    let mut screen_rotation = true;
    let mut debug_zoom = 1.0;

    let mut car = Car {
        length: 7.,
        width: 5.,
        speed: 0.0,
        rotation: 0.0,
        pos: vec2(0., 0.),
        color: DARKPURPLE,
        braking: false,
    };
    loop {
        car.update(Actions {
            accelerate: is_key_down(KeyCode::W),
            reverse: is_key_down(KeyCode::S),
            brake: is_key_down(KeyCode::Space),
            left: is_key_down(KeyCode::A),
            right: is_key_down(KeyCode::D),
        });
        if is_key_pressed(KeyCode::Up) {
            debug_zoom += 1.0;
        }
        if is_key_pressed(KeyCode::Down) {
            debug_zoom -= 1.0;
        }

        if screen_rotation {
            smooth_cam_rotation = smooth_cam_rotation * 0.99 + car.rotation * 0.01;
            camera.rotation = -smooth_cam_rotation.to_degrees() - 90.;
        }
        if is_key_pressed(KeyCode::Key1) {
            screen_rotation = !screen_rotation;
        }
        smooth_cam_speed = smooth_cam_speed * 0.99 + car.speed * 0.01;
        smooth_cam_speed = smooth_cam_speed.clamp(0.0, 3.0);
        camera.zoom = standard_zoom * (3.1 - smooth_cam_speed);
        camera.zoom /= debug_zoom;
        set_camera(&camera);
        camera.zoom *= debug_zoom;

        // Avoid moving everything in whole pixels and allow for smooth sub-pixel movement instead
        let adjust = -car.pos.fract();

        let player_pos = Point2d {
            x: car.pos.x as i64,
            y: car.pos.y as i64,
        };
        let load_time = time::get_time();
        let load_time = ((time::get_time() - load_time) * 10000.).round() / 10.;
        if load_time > 0. {
            last_load_time = load_time;
        }

        clear_background(DARKGREEN);

        let point2screen = |point: Point2d| -> Vec2 {
            let point = point - player_pos;
            i64vec2(point.x, point.y).as_vec2() + adjust
        };

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

        // TODO: make the vision range calculation robust for arbitrary algorithms.
        let padding = camera.screen_to_world(Vec2::splat(0.)).length();
        draw_circle_lines(0., 0., padding, debug_zoom, PURPLE);
        let padding = Point2d::splat(padding as i64);
        let mut vision_range = Bounds::point(player_pos).pad(padding);
        draw_bounds(vision_range);
        let padding = i64::from(RoadsChunk::SIZE.x.get());
        vision_range.min.x -= padding;
        vision_range.min.y -= padding;
        draw_bounds(vision_range);

        for index in Bounds::point(RoadsChunk::pos_to_grid(player_pos))
            .pad(Point2d::splat(GridIndex(1)))
            .iter()
        {
            let current_chunk = RoadsChunk::bounds(index);
            draw_bounds(current_chunk);
        }

        for roads in player.roads.get_range(vision_range) {
            for &line in roads.roads.iter() {
                let start = point2screen(line.start);
                let end = point2screen(line.end);
                draw_line(start.x, start.y, end.x, end.y, 40., GRAY);
                draw_circle(start.x, start.y, 20., GRAY);
                draw_circle(start.x, start.y, 2., WHITE);
                draw_circle(end.x, end.y, 20., GRAY);
                draw_circle(end.x, end.y, 2., WHITE);
            }
        }
        for roads in player.roads.get_range(vision_range) {
            for &line in roads.roads.iter() {
                let start = point2screen(line.start);
                let end = point2screen(line.end);
                draw_line(start.x, start.y, end.x, end.y, 4., WHITE);
            }
        }
        car.draw();

        set_camera(&overlay_camera);
        draw_text(
            &format!("last load time: {last_load_time}ms"),
            0.,
            10.,
            10.,
            WHITE,
        );

        next_frame().await
    }
}

struct Car {
    length: f32,
    width: f32,
    rotation: f32,
    color: Color,
    speed: f32,
    pos: Vec2,
    /// Used to ensure that braking doesn't go into reversing without releasing and
    /// repressing the key.
    braking: bool,
}

struct Actions {
    accelerate: bool,
    brake: bool,
    reverse: bool,
    left: bool,
    right: bool,
}

impl Car {
    fn update(&mut self, actions: Actions) {
        let braking = self.braking || actions.brake || self.speed > 0. && actions.reverse;
        self.braking = actions.brake;
        if braking {
            if self.speed > 0. {
                self.speed = (self.speed - 0.1).clamp(0.0, 2.0);
            } else {
                self.speed = (self.speed + 0.1).clamp(-0.3, 0.0);
            }
        } else if actions.reverse {
            self.speed -= 0.01;
        } else if actions.accelerate {
            self.speed += 0.01;
        } else {
            self.speed *= 0.99;
        }

        if actions.left {
            self.rotation -= f32::to_radians(1.) * self.speed;
        }
        if actions.right {
            self.rotation += f32::to_radians(1.) * self.speed;
        }
        self.speed = self.speed.clamp(-0.3, 2.0);
        if is_key_down(KeyCode::LeftShift) {
            self.speed *= 10.;
        }
        self.pos += Vec2::from_angle(self.rotation) * self.speed;
    }
    fn draw(&self) {
        draw_rectangle_ex(
            0.,
            0.,
            self.length,
            self.width,
            DrawRectangleParams {
                offset: vec2(0.5, 0.5),
                rotation: self.rotation,
                color: self.color,
            },
        );
        let rotation = Vec2::from_angle(self.rotation) * self.length / 2.;
        draw_circle(rotation.x, rotation.y, self.width / 2., self.color);

        if self.braking || self.speed < 0. {
            let rotation = Vec2::from_angle(self.rotation) * (self.length / 2. + 1.);
            draw_rectangle_ex(
                -rotation.x,
                -rotation.y,
                2.,
                self.width,
                DrawRectangleParams {
                    offset: vec2(0.5, 0.5),
                    rotation: self.rotation,
                    color: if self.speed < 0. { WHITE } else { RED },
                },
            );
        }
    }
}
