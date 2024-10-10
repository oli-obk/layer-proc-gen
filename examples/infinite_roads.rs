use ::rand::prelude::*;
use ::tracing::{debug, trace};
use macroquad::prelude::*;
use std::{f32::consts::PI, sync::Arc};

use layer_proc_gen::*;
use rolling_grid::{GridIndex, GridPoint, RollingGrid};
use vec2::{GridBounds, Point2d};

#[path = "../tests/tracing.rs"]
mod tracing_helper;
use tracing_helper::*;

#[derive(Default)]
struct Locations(RollingGrid<Self>);
#[derive(PartialEq, Debug)]
struct LocationsChunk {
    points: [Point2d; 3],
}

impl Layer for Locations {
    type Chunk = LocationsChunk;

    fn rolling_grid(&self) -> &RollingGrid<Self> {
        &self.0
    }

    fn ensure_all_deps(&self, _chunk_bounds: GridBounds) {}
}

impl Chunk for LocationsChunk {
    type Layer = Locations;

    fn compute(_layer: &Self::Layer, index: GridPoint) -> Self {
        let chunk_bounds = Self::bounds(index);
        trace!(?chunk_bounds);
        let mut seed = [0; 32];
        seed[0..16].copy_from_slice(&index.map(|GridIndex(i)| i).to_ne_bytes());
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

struct Roads {
    grid: RollingGrid<Self>,
    locations: LayerDependency<Locations, 256, 256>,
}

#[derive(PartialEq, Debug)]
struct RoadsChunk {
    roads: Vec<(Point2d, Point2d)>,
}

impl Layer for Roads {
    type Chunk = RoadsChunk;

    fn rolling_grid(&self) -> &RollingGrid<Self> {
        &self.grid
    }

    #[track_caller]
    fn ensure_all_deps(&self, chunk_bounds: GridBounds) {
        self.locations.ensure_loaded_in_bounds(chunk_bounds);
    }
}

impl Chunk for RoadsChunk {
    type Layer = Roads;

    fn compute(layer: &Self::Layer, index: GridPoint) -> Self {
        let mut seed = [0; 32];
        seed[0..16].copy_from_slice(&index.map(|GridIndex(i)| i).to_ne_bytes());
        let mut rng = SmallRng::from_seed(seed);
        let mut roads = vec![];
        for location in layer.locations.get(index).points {
            trace!(?location);
            let mut possible_destinations: Vec<_> = layer
                .locations
                .get_grid_range(GridBounds {
                    min: index,
                    max: index + Point2d { x: 1, y: 1 }.map(GridIndex),
                })
                .flat_map(|grid| grid.points.into_iter())
                .filter(|point| point.x > location.x || point.y > location.y)
                .collect();
            possible_destinations.shuffle(&mut rng);
            for _ in 0..rng.gen_range(0..3) {
                if let Some(dest) = possible_destinations.pop() {
                    roads.push((location, dest));
                }
            }
        }
        debug!(?roads);
        RoadsChunk { roads }
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

#[derive(PartialEq, Debug)]
struct PlayerChunk;

impl Layer for Player {
    type Chunk = PlayerChunk;

    fn rolling_grid(&self) -> &RollingGrid<Self> {
        &self.grid
    }

    const GRID_SIZE: Point2d<u8> = Point2d::splat(3);

    const GRID_OVERLAP: u8 = 2;

    fn ensure_all_deps(&self, chunk_bounds: GridBounds) {
        self.roads.ensure_loaded_in_bounds(chunk_bounds);
    }
}

impl Chunk for PlayerChunk {
    type Layer = Player;

    fn compute(_layer: &Self::Layer, _index: GridPoint) -> Self {
        PlayerChunk
    }
}

#[macroquad::main("layer proc gen demo")]
async fn main() {
    init_tracing();
    let locations = Arc::new(Locations::default());
    let roads = Arc::new(Roads {
        grid: Default::default(),
        locations: locations.into(),
    });
    let player = Player::new(roads.clone());
    let mut player_pos = Vec2::new(0., 0.);
    let mut rotation = 0.0;
    let mut speed: f32 = 0.0;
    loop {
        if is_key_down(KeyCode::W) {
            speed += 0.01;
        }
        if is_key_down(KeyCode::A) {
            rotation -= PI / 180.;
        }
        if is_key_down(KeyCode::S) {
            speed -= 0.1;
        }
        if is_key_down(KeyCode::D) {
            rotation += PI / 180.;
        }
        speed = speed.clamp(0.0, 2.0);
        player_pos += Vec2::from_angle(rotation) * speed;

        let screen_center = vec2(screen_width(), screen_height()) / 2.;
        // Avoid moving everything in whole pixels and allow for smooth sub-pixel movement instead
        let center = screen_center - player_pos.fract();

        let player_pos = Point2d {
            x: player_pos.x as i64,
            y: player_pos.y as i64,
        };
        player.ensure_loaded_in_bounds(GridBounds::point(player_pos));

        clear_background(DARKGREEN);

        let point2screen = |point: Point2d| -> Vec2 {
            let point = point - player_pos;
            i64vec2(point.x, point.y).as_vec2() + center
        };

        let vision_range = GridBounds::point(player_pos).pad(player.roads.padding());
        trace!(?vision_range);
        for roads in player.roads.get_range(vision_range) {
            for &(start, end) in roads.roads.iter() {
                let start = point2screen(start);
                let end = point2screen(end);
                draw_line(start.x, start.y, end.x, end.y, 35., GRAY);
            }
        }
        for roads in player.roads.get_range(vision_range) {
            for &(start, end) in roads.roads.iter() {
                let start = point2screen(start);
                let end = point2screen(end);
                draw_line(start.x, start.y, end.x, end.y, 4., WHITE);
                draw_circle(start.x, start.y, 20., BLUE);
                draw_circle(end.x, end.y, 20., BLUE);
            }
        }
        draw_rectangle_ex(
            screen_center.x,
            screen_center.y,
            10.0,
            10.0,
            DrawRectangleParams {
                offset: vec2(0.5, 0.5),
                rotation,
                color: RED,
            },
        );

        draw_text(&speed.to_string(), 0., 10., 10., BLACK);

        next_frame().await
    }
}
