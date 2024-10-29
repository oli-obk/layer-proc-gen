use std::sync::Arc;

use layer_proc_gen::*;
use rolling_grid::{GridIndex, GridPoint, RollingGrid};
use vec2::{Bounds, Point2d};

mod tracing;
use tracing::*;

#[derive(Default)]
struct TheLayer(RollingGrid<Self>);
#[expect(dead_code)]
#[derive(Clone, Default)]
struct TheChunk(usize);

impl Layer for TheLayer {
    type Chunk = TheChunk;
    type Store = Arc<Self>;

    fn rolling_grid(&self) -> &RollingGrid<Self> {
        &self.0
    }

    fn ensure_all_deps(&self, _chunk_bounds: Bounds) {}
}

impl Chunk for TheChunk {
    type Layer = TheLayer;
    type Store = Self;

    fn compute(_layer: &Self::Layer, _index: GridPoint<Self>) -> Self {
        TheChunk(0)
    }
}

struct Player {
    grid: RollingGrid<Self>,
    the_layer: LayerDependency<TheLayer>,
}

impl Player {
    pub fn new(the_layer: LayerDependency<TheLayer>) -> Self {
        Self {
            grid: Default::default(),
            the_layer,
        }
    }
}

#[derive(Clone, Default)]
struct PlayerChunk;

impl Layer for Player {
    type Chunk = PlayerChunk;
    type Store = Self;

    fn rolling_grid(&self) -> &RollingGrid<Self> {
        &self.grid
    }

    const GRID_SIZE: Point2d<u8> = Point2d::splat(0);

    const GRID_OVERLAP: u8 = 1;

    fn ensure_all_deps(&self, chunk_bounds: Bounds) {
        self.the_layer.ensure_loaded_in_bounds(chunk_bounds);
    }
}

impl Chunk for PlayerChunk {
    type Layer = Player;
    type Store = Self;

    const SIZE: Point2d<u8> = Point2d::splat(0);

    fn compute(_layer: &Self::Layer, _index: GridPoint<Self>) -> Self {
        PlayerChunk
    }
}

struct Map {
    grid: RollingGrid<Self>,
    the_layer: LayerDependency<TheLayer>,
}

impl Map {
    pub fn new(the_layer: LayerDependency<TheLayer>) -> Self {
        Self {
            grid: Default::default(),
            the_layer,
        }
    }
}

#[derive(Clone, Default)]
struct MapChunk;

impl Layer for Map {
    type Chunk = MapChunk;
    type Store = Self;

    fn rolling_grid(&self) -> &RollingGrid<Self> {
        &self.grid
    }

    fn ensure_all_deps(&self, chunk_bounds: Bounds) {
        self.the_layer.ensure_loaded_in_bounds(chunk_bounds);
    }

    const GRID_SIZE: Point2d<u8> = Point2d::splat(0);

    const GRID_OVERLAP: u8 = 1;
}

impl Chunk for MapChunk {
    type Layer = Map;
    type Store = Self;

    const SIZE: Point2d<u8> = Point2d::splat(0);

    fn compute(_layer: &Self::Layer, _index: GridPoint<Self>) -> Self {
        MapChunk
    }
}

#[test]
fn create_layer() {
    let layer = TheLayer::default();
    layer.rolling_grid().get_or_compute(
        Point2d { x: 42, y: 99 }.map(GridIndex::<TheChunk>::from_raw),
        &layer,
    );
}

#[test]
fn double_assign_chunk() {
    let layer = TheLayer::default();
    layer.rolling_grid().get_or_compute(
        Point2d { x: 42, y: 99 }.map(GridIndex::<TheChunk>::from_raw),
        &layer,
    );
    // This is very incorrect, but adding assertions for checking its
    // correctness destroys all caching and makes logging and perf
    // completely useless.
    layer.rolling_grid().get_or_compute(
        Point2d { x: 42, y: 99 }.map(GridIndex::<TheChunk>::from_raw),
        &layer,
    );
}

#[test]
fn create_player() {
    init_tracing();
    let the_layer = TheLayer::new();
    let player = Player::new(the_layer.clone()).into_dep();
    let player_pos = Point2d { x: 42, y: 99 };
    player.ensure_loaded_in_bounds(Bounds::point(player_pos));
    let map = Map::new(the_layer).into_dep();
    map.ensure_loaded_in_bounds(Bounds::point(player_pos));
}
