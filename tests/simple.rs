use std::{num::NonZeroU16, sync::Arc};

use layer_proc_gen::*;
use rolling_grid::RollingGrid;
use vec2::{GridBounds, Point2d};

fn init_tracing() {
    use tracing_subscriber::layer::SubscriberExt as _;
    let subscriber = tracing_subscriber::Registry::default().with(
        tracing_tree::HierarchicalLayer::new(2)
            .with_indent_lines(true)
            .with_bracketed_fields(true)
            .with_targets(true),
    );
    tracing::subscriber::set_global_default(subscriber).unwrap();
    eprintln!();
}

#[derive(Default)]
struct TheLayer(RollingGrid<Self>);
#[expect(dead_code)]
struct TheChunk(usize);

impl Layer for TheLayer {
    type Chunk = TheChunk;

    fn rolling_grid(&self) -> &RollingGrid<Self> {
        &self.0
    }

    fn ensure_all_deps(&self, _chunk_bounds: GridBounds) {}
}

impl Chunk for TheChunk {
    type Layer = TheLayer;

    fn compute(_layer: &Self::Layer, _index: Point2d) -> Self {
        TheChunk(0)
    }
}

struct Player {
    grid: RollingGrid<Self>,
    the_layer: LayerDependency<TheLayer, 16, 16>,
}

impl Player {
    pub fn new(the_layer: Arc<TheLayer>) -> Self {
        Self {
            grid: Default::default(),
            the_layer: the_layer.into(),
        }
    }
}

struct PlayerChunk;

impl Layer for Player {
    type Chunk = PlayerChunk;

    fn rolling_grid(&self) -> &RollingGrid<Self> {
        &self.grid
    }

    const GRID_SIZE: Point2d<u8> = Point2d::splat(1);

    const GRID_OVERLAP: u8 = 1;

    fn ensure_all_deps(&self, chunk_bounds: GridBounds) {
        self.the_layer.ensure_loaded_in_bounds(chunk_bounds);
    }
}

impl Chunk for PlayerChunk {
    type Layer = Player;

    const SIZE: Point2d<NonZeroU16> = match NonZeroU16::new(1) {
        Some(v) => Point2d::splat(v),
        None => std::unreachable!(),
    };

    fn compute(_layer: &Self::Layer, _index: Point2d) -> Self {
        PlayerChunk
    }
}

struct Map {
    grid: RollingGrid<Self>,
    the_layer: LayerDependency<TheLayer, 100, 100>,
}

impl Map {
    pub fn new(the_layer: Arc<TheLayer>) -> Self {
        Self {
            grid: Default::default(),
            the_layer: the_layer.into(),
        }
    }
}

struct MapChunk;

impl Layer for Map {
    type Chunk = MapChunk;

    fn rolling_grid(&self) -> &RollingGrid<Self> {
        &self.grid
    }

    fn ensure_all_deps(&self, chunk_bounds: GridBounds) {
        self.the_layer.ensure_loaded_in_bounds(chunk_bounds);
    }

    const GRID_SIZE: Point2d<u8> = Point2d::splat(1);

    const GRID_OVERLAP: u8 = 1;
}

impl Chunk for MapChunk {
    type Layer = Map;

    const SIZE: Point2d<NonZeroU16> = match NonZeroU16::new(1) {
        Some(v) => Point2d::splat(v),
        None => std::unreachable!(),
    };

    fn compute(_layer: &Self::Layer, _index: Point2d) -> Self {
        MapChunk
    }
}

#[test]
fn create_layer() {
    let layer = TheLayer::default();
    layer
        .rolling_grid()
        .set(Point2d { x: 42, y: 99 }, TheChunk(0));
}

#[test]
#[should_panic]
fn double_assign_chunk() {
    let layer = TheLayer::default();
    layer
        .rolling_grid()
        .set(Point2d { x: 42, y: 99 }, TheChunk(0));
    layer
        .rolling_grid()
        .set(Point2d { x: 42, y: 99 }, TheChunk(1));
}

#[test]
fn create_player() {
    init_tracing();
    let the_layer = Arc::new(TheLayer::default());
    let player = Player::new(the_layer.clone());
    let player_pos = Point2d { x: 42, y: 99 };
    player.ensure_loaded_in_bounds(GridBounds::point(player_pos));
    let map = Map::new(the_layer);
    map.ensure_loaded_in_bounds(GridBounds::point(player_pos));
}
