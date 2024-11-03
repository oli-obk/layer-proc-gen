use std::sync::Arc;

use layer_proc_gen::*;
use rolling_grid::{GridIndex, GridPoint};
use vec2::{Bounds, Point2d};

mod tracing;
use tracing::*;

#[expect(dead_code)]
#[derive(Clone, Default)]
struct TheChunk(usize);

impl Chunk for TheChunk {
    type LayerStore<T> = Arc<T>;
    type Layer = ();
    type Store = Self;

    fn compute(_layer: &Self::Layer, _index: GridPoint<Self>) -> Self {
        TheChunk(0)
    }
}

#[derive(Clone, Default)]
struct PlayerChunk;

impl Chunk for PlayerChunk {
    type LayerStore<T> = T;
    type Layer = (LayerDependency<TheChunk>,);
    type Store = Self;

    const GRID_SIZE: Point2d<u8> = Point2d::splat(0);

    const GRID_OVERLAP: u8 = 1;

    const SIZE: Point2d<u8> = Point2d::splat(0);

    fn compute((layer,): &Self::Layer, index: GridPoint<Self>) -> Self {
        for _ in layer.get_range(Self::bounds(index)) {}
        PlayerChunk
    }
}

#[derive(Clone, Default)]
struct MapChunk;

impl Chunk for MapChunk {
    type LayerStore<T> = T;
    type Layer = (LayerDependency<TheChunk>,);
    type Store = Self;

    const SIZE: Point2d<u8> = Point2d::splat(0);

    const GRID_SIZE: Point2d<u8> = Point2d::splat(0);

    const GRID_OVERLAP: u8 = 1;

    fn compute((layer,): &Self::Layer, index: GridPoint<Self>) -> Self {
        for _ in layer.get_range(Self::bounds(index)) {}
        MapChunk
    }
}

#[test]
fn create_layer() {
    let layer = LayerDependency::from(());
    layer.get_or_compute(Point2d { x: 42, y: 99 }.map(GridIndex::<TheChunk>::from_raw));
}

#[test]
fn double_assign_chunk() {
    let layer = LayerDependency::from(());
    layer.get_or_compute(Point2d { x: 42, y: 99 }.map(GridIndex::<TheChunk>::from_raw));
    // This is very incorrect, but adding assertions for checking its
    // correctness destroys all caching and makes logging and perf
    // completely useless.
    layer.get_or_compute(Point2d { x: 42, y: 99 }.map(GridIndex::<TheChunk>::from_raw));
}

#[test]
fn create_player() {
    init_tracing();
    let the_layer = LayerDependency::from(());
    let player = LayerDependency::<PlayerChunk>::from((the_layer.clone(),));
    let player_pos = Point2d { x: 42, y: 99 };
    player.ensure_loaded_in_bounds(Bounds::point(player_pos));
    let map = LayerDependency::<MapChunk>::from((the_layer,));
    map.ensure_loaded_in_bounds(Bounds::point(player_pos));
}
