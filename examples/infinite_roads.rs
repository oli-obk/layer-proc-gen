use ::rand::prelude::*;
use ::tracing::{debug, trace};
use arrayvec::ArrayVec;
use macroquad::prelude::*;
use miniquad::window::screen_size;
use std::{
    borrow::Borrow,
    cell::{Cell, Ref, RefCell},
    num::{NonZeroU16, NonZeroU8},
    sync::Arc,
};

use layer_proc_gen::*;
use rolling_grid::{GridIndex, GridPoint, RollingGrid};
use vec2::{Bounds, Line, Num, Point2d};

#[path = "../tests/tracing.rs"]
mod tracing_helper;
use tracing_helper::*;

#[derive(Default)]
struct Cities(RollingGrid<Self>);
#[derive(PartialEq, Debug, Clone, Default)]
struct CitiesChunk {
    points: [Point2d; 3],
}

impl Layer for Cities {
    type Chunk = CitiesChunk;

    fn rolling_grid(&self) -> &RollingGrid<Self> {
        &self.0
    }

    fn ensure_all_deps(&self, _chunk_bounds: Bounds) {}
}

impl Chunk for CitiesChunk {
    type Layer = Cities;
    type Store = Self;

    const SIZE: Point2d<NonZeroU16> = match NonZeroU16::new(256 * 32) {
        Some(v) => Point2d::splat(v),
        None => unreachable!(),
    };

    fn compute(_layer: &Self::Layer, index: GridPoint<Self>) -> Self {
        Self {
            points: generate_points::<Self, 3>(index, 1),
        }
    }
}

#[derive(Default)]
struct Locations(RollingGrid<Self>);
#[derive(PartialEq, Debug, Clone, Default)]
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

    fn compute(_layer: &Self::Layer, index: GridPoint<Self>) -> Self {
        LocationsChunk {
            points: generate_points::<Self, 3>(index, 0),
        }
    }
}

fn generate_points<C: Chunk + 'static, const N: usize>(
    index: GridPoint<C>,
    salt: u64,
) -> [Point2d; N] {
    let chunk_bounds = C::bounds(index);
    trace!(?chunk_bounds);
    let x = SmallRng::seed_from_u64(index.x.0 as u64);
    let y = SmallRng::seed_from_u64(index.y.0 as u64);
    let salt = SmallRng::seed_from_u64(salt);
    let mut seed = [0; 32];
    for mut rng in [x, y, salt] {
        let mut tmp = [0; 32];
        rng.fill_bytes(&mut tmp);
        for (seed, tmp) in seed.iter_mut().zip(tmp.iter()) {
            *seed ^= *tmp;
        }
    }
    let mut rng = SmallRng::from_seed(seed);
    std::array::from_fn(|_| chunk_bounds.sample(&mut rng))
}

/// Removes locations that are too close to others
struct ReducedLocations {
    grid: RollingGrid<Self>,
    raw_locations: LayerDependency<Locations, 0, 0>,
    cities: LayerDependency<Cities, 0, 0>,
}

#[derive(PartialEq, Debug, Clone, Default)]
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
        self.cities.ensure_loaded_in_bounds(chunk_bounds);
    }
}

impl Chunk for ReducedLocationsChunk {
    type Layer = ReducedLocations;
    type Store = Self;

    fn compute(layer: &Self::Layer, index: GridPoint<Self>) -> Self {
        let bounds = Self::bounds(index);
        let center = bounds.center();
        if layer.cities.get_range(bounds).all(|cities| {
            cities
                .points
                .iter()
                .all(|&city| center.dist_squared(city) > 1000 * 1000)
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

#[derive(PartialEq, Debug, Default)]
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

    fn compute(layer: &Self::Layer, index: GridPoint<Self>) -> Self::Store {
        let roads = gen_roads(
            layer
                .locations
                .get_grid_range(
                    Bounds::point(index.into_same_chunk_size()).pad(Point2d::splat(GridIndex::ONE)),
                )
                .map(|chunk| chunk.points),
        );
        RoadsChunk { roads }.into()
    }
}

fn gen_roads(chunks: impl Iterator<Item = impl Borrow<[Point2d]>>) -> Vec<Line> {
    let mut roads = vec![];
    let mut points: ArrayVec<Point2d, { 3 * 9 }> = ArrayVec::new();
    let mut start = usize::MAX;
    let mut n = usize::MAX;
    for (i, grid) in chunks.enumerate() {
        let grid = grid.borrow();
        if i == 4 {
            start = points.len();
            n = grid.len();
        }
        points.extend(grid.iter().copied());
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
    roads
}

struct Highways {
    grid: RollingGrid<Self>,
    cities: LayerDependency<Cities, 256, 256>,
}

#[derive(PartialEq, Debug, Default)]
struct HighwaysChunk {
    roads: Vec<Line>,
}

impl Layer for Highways {
    type Chunk = HighwaysChunk;

    fn rolling_grid(&self) -> &RollingGrid<Self> {
        &self.grid
    }

    #[track_caller]
    fn ensure_all_deps(&self, chunk_bounds: Bounds) {
        self.cities.ensure_loaded_in_bounds(chunk_bounds);
    }
}

impl Chunk for HighwaysChunk {
    type Layer = Highways;
    type Store = Arc<Self>;
    const SIZE: Point2d<NonZeroU16> = CitiesChunk::SIZE;

    fn compute(layer: &Self::Layer, index: GridPoint<Self>) -> Self::Store {
        let roads = gen_roads(
            layer
                .cities
                .get_grid_range(
                    Bounds::point(index.into_same_chunk_size()).pad(Point2d::splat(GridIndex::ONE)),
                )
                .map(|chunk| chunk.points),
        );
        HighwaysChunk { roads }.into()
    }
}

struct Player {
    roads: LayerDependency<Roads, 1000, 1000>,
    highways: LayerDependency<Highways, 1000, 1000>,
    max_zoom_in: NonZeroU8,
    max_zoom_out: NonZeroU8,
    car: Car,
    last_grid_vision_range: Cell<(
        Bounds<GridIndex<RoadsChunk>>,
        Bounds<GridIndex<HighwaysChunk>>,
    )>,
    roads_for_last_grid_vision_range: RefCell<Vec<Line>>,
}

impl Player {
    pub fn new(roads: Arc<Roads>, highways: Arc<Highways>) -> Self {
        Self {
            roads: roads.into(),
            highways: highways.into(),
            max_zoom_in: NonZeroU8::new(3).unwrap(),
            max_zoom_out: NonZeroU8::new(10).unwrap(),
            car: Car {
                length: 7.,
                width: 5.,
                speed: 0.0,
                rotation: 0.0,
                pos: vec2(1200., 300.),
                color: DARKPURPLE,
                braking: false,
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
        let adjust = self.car.pos.fract();
        move |point: Point2d| -> Vec2 {
            let point = point - player_pos;
            i64vec2(point.x, point.y).as_vec2() - adjust
        }
    }

    fn pos(&self) -> Point2d {
        let player_pos = Point2d {
            x: self.car.pos.x as i64,
            y: self.car.pos.y as i64,
        };
        player_pos
    }

    pub fn vision_range<C: Chunk>(&self, half_screen_visible_area: Vec2) -> Bounds {
        let padding = half_screen_visible_area.abs().ceil().as_i64vec2();
        let padding = Point2d::new(padding.x as i64, padding.y as i64);
        let mut vision_range = Bounds::point(self.pos()).pad(padding);
        let padding = C::SIZE.map(|p| i64::from(p.get()));
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

    pub fn roads(&self, half_screen_visible_area: Vec2) -> Ref<'_, [Line]> {
        let grid_vision_range = self.grid_vision_range(half_screen_visible_area);
        let highway_vision_range = self.grid_vision_range(half_screen_visible_area);
        if (grid_vision_range, highway_vision_range) != self.last_grid_vision_range.get() {
            self.last_grid_vision_range
                .set((grid_vision_range, highway_vision_range));
            let mut roads = self.roads_for_last_grid_vision_range.borrow_mut();
            roads.clear();
            for index in grid_vision_range.iter() {
                roads.extend_from_slice(&self.roads.get_or_compute(index).roads);
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

    let raw_locations = Arc::new(Locations::default());
    let cities = Arc::new(Cities::default());
    let locations = Arc::new(ReducedLocations {
        grid: Default::default(),
        raw_locations: raw_locations.into(),
        cities: cities.clone().into(),
    });
    let roads = Arc::new(Roads {
        grid: Default::default(),
        locations: locations.clone().into(),
    });
    let highways = Arc::new(Highways {
        grid: Default::default(),
        cities: cities.into(),
    });
    let mut player = Player::new(roads, highways);
    let mut smooth_cam_speed = 0.0;
    let mut debug_zoom = 1.0;

    let start = locations
        .cities
        .get_range(Bounds::point(player.pos()))
        .next()
        .unwrap()
        .points[0];
    player.car.pos.x = start.x as _;
    player.car.pos.y = start.y as _;

    loop {
        player.car.update(Actions {
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

        smooth_cam_speed = smooth_cam_speed * 0.99 + player.car.speed * 0.01;
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

        // TODO: make the vision range calculation robust for arbitrary algorithms.
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

        for &line in player.roads(padding).iter() {
            let start = point2screen(line.start);
            let end = point2screen(line.end);
            draw_line(line, 40., GRAY);
            draw_circle(start.x, start.y, 20., GRAY);
            draw_circle(start.x, start.y, 2., WHITE);
            draw_circle(end.x, end.y, 20., GRAY);
            draw_circle(end.x, end.y, 2., WHITE);
        }
        for &line in player.roads(padding).iter() {
            draw_line(line, 4., WHITE);
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
        }

        player.car.draw();

        set_camera(&overlay_camera);
        draw_text(&format!("fps: {}", get_fps()), 0., 30., 30., WHITE);
        draw_text(&format!("pos: {:?}", player.car.pos), 0., 60., 30., WHITE);

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
