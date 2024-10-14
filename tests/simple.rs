use std::{
    future::Future,
    num::NonZeroU16,
    pin::pin,
    sync::Arc,
    task::{Context, Wake},
};

use layer_proc_gen::*;
use rolling_grid::{GridIndex, GridPoint, RollingGrid};
use vec2::{Bounds, Point2d};

mod tracing;
use tracing::*;

#[derive(Default)]
struct TheLayer(RollingGrid<Self>);
#[expect(dead_code)]
struct TheChunk(usize);

impl Layer for TheLayer {
    type Chunk = TheChunk;

    fn rolling_grid(&self) -> &RollingGrid<Self> {
        &self.0
    }

    async fn ensure_all_deps(&self, _chunk_bounds: Bounds) {}
}

impl Chunk for TheChunk {
    type Layer = TheLayer;

    async fn compute(_layer: &Self::Layer, _index: GridPoint) -> Self {
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

    async fn ensure_all_deps(&self, chunk_bounds: Bounds) {
        self.the_layer.ensure_loaded_in_bounds(chunk_bounds).await;
    }
}

impl Chunk for PlayerChunk {
    type Layer = Player;

    const SIZE: Point2d<NonZeroU16> = match NonZeroU16::new(1) {
        Some(v) => Point2d::splat(v),
        None => std::unreachable!(),
    };

    async fn compute(_layer: &Self::Layer, _index: GridPoint) -> Self {
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

    async fn ensure_all_deps(&self, chunk_bounds: Bounds) {
        self.the_layer.ensure_loaded_in_bounds(chunk_bounds).await;
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

    async fn compute(_layer: &Self::Layer, _index: GridPoint) -> Self {
        MapChunk
    }
}

#[track_caller]
fn expect_future_doesnt_pend<T>(f: impl Future<Output = T>) -> T {
    let f = pin!(f);

    struct Dummy;
    impl Wake for Dummy {
        fn wake(self: Arc<Self>) {
            todo!()
        }
    }
    match f.poll(&mut Context::from_waker(&Arc::new(Dummy).into())) {
        std::task::Poll::Ready(v) => v,
        std::task::Poll::Pending => panic!("future pended"),
    }
}

#[test]
fn create_layer() {
    let layer = TheLayer::default();
    let fut = layer
        .rolling_grid()
        .set(Point2d { x: 42, y: 99 }.map(GridIndex), || {
            std::future::ready(TheChunk(0))
        });
    expect_future_doesnt_pend(fut);
}

#[test]
fn double_assign_chunk() {
    let layer = TheLayer::default();
    let fut = layer
        .rolling_grid()
        .set(Point2d { x: 42, y: 99 }.map(GridIndex), || {
            std::future::ready(TheChunk(0))
        });
    expect_future_doesnt_pend(fut);
    // This is very incorrect, but adding assertions for checking its
    // correctness destroys all caching and makes logging and perf
    // completely useless.
    let fut = layer
        .rolling_grid()
        .set(Point2d { x: 42, y: 99 }.map(GridIndex), || {
            std::future::ready(TheChunk(0))
        });
    expect_future_doesnt_pend(fut);
}

#[test]
fn create_player() {
    init_tracing();
    let the_layer = Arc::new(TheLayer::default());
    let player = Player::new(the_layer.clone());
    let player_pos = Point2d { x: 42, y: 99 };
    expect_future_doesnt_pend(player.ensure_loaded_in_bounds(Bounds::point(player_pos)));
    let map = Map::new(the_layer);
    expect_future_doesnt_pend(map.ensure_loaded_in_bounds(Bounds::point(player_pos)));
}
