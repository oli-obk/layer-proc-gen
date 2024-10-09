use std::sync::Arc;

use layer_proc_gen::*;
use rolling_grid::RollingGrid;
use vec2::Point2d;

#[derive(Default)]
struct TheLayer(RollingGrid<Self>);
struct TheChunk;

impl Layer for TheLayer {
    type Chunk = TheChunk;

    fn rolling_grid(&self) -> &RollingGrid<Self> {
        &self.0
    }

    fn rolling_grid_mut(&mut self) -> &mut RollingGrid<Self> {
        &mut self.0
    }
}

impl Chunk for TheChunk {
    type Layer = TheLayer;
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

    fn rolling_grid_mut(&mut self) -> &mut RollingGrid<Self> {
        &mut self.grid
    }

    const GRID_HEIGHT: usize = 1;

    const GRID_WIDTH: usize = 1;

    const GRID_OVERLAP: usize = 1;
}

impl Chunk for PlayerChunk {
    type Layer = Player;

    const WIDTH: std::num::NonZeroUsize = match std::num::NonZeroUsize::new(1) {
        Some(v) => v,
        None => std::unreachable!(),
    };

    const HEIGHT: std::num::NonZeroUsize = match std::num::NonZeroUsize::new(1) {
        Some(v) => v,
        None => std::unreachable!(),
    };
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

    fn rolling_grid_mut(&mut self) -> &mut RollingGrid<Self> {
        &mut self.grid
    }

    const GRID_HEIGHT: usize = 1;

    const GRID_WIDTH: usize = 1;

    const GRID_OVERLAP: usize = 1;
}

impl Chunk for MapChunk {
    type Layer = Map;

    const WIDTH: std::num::NonZeroUsize = match std::num::NonZeroUsize::new(1) {
        Some(v) => v,
        None => std::unreachable!(),
    };

    const HEIGHT: std::num::NonZeroUsize = match std::num::NonZeroUsize::new(1) {
        Some(v) => v,
        None => std::unreachable!(),
    };
}

#[test]
fn create_layer() {
    let mut layer = TheLayer::default();
    layer
        .rolling_grid_mut()
        .set(Point2d { x: 42, y: 99 }, TheChunk);
}

#[test]
#[should_panic]
fn double_assign_chunk() {
    let mut layer = TheLayer::default();
    layer
        .rolling_grid_mut()
        .set(Point2d { x: 42, y: 99 }, TheChunk);
    layer
        .rolling_grid_mut()
        .set(Point2d { x: 42, y: 99 }, TheChunk);
}

#[test]
fn create_player() {
    let the_layer = Arc::new(TheLayer::default());
    let mut layer = Player::new(the_layer.clone());
    layer
        .rolling_grid_mut()
        .set(Point2d { x: 42, y: 99 }, PlayerChunk);
    let mut layer = Map::new(the_layer);
    layer
        .rolling_grid_mut()
        .set(Point2d { x: 42, y: 99 }, MapChunk);
}
