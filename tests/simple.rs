use std::sync::Arc;

use layer_proc_gen::*;
use vec2::{Bounds, Point2d};

mod tracing;
use tracing::*;

#[expect(dead_code)]
#[derive(Clone, Default)]
struct TheChunk(usize);

impl Chunk for TheChunk {
    type LayerStore<T> = Arc<T>;
    type Dependencies = ();

    fn compute(_layer: &Self::Dependencies, _index: GridPoint<Self>) -> Self {
        TheChunk(0)
    }
}

#[derive(Clone, Default)]
struct Player;

impl Chunk for Player {
    type LayerStore<T> = T;
    type Dependencies = (TheChunk,);

    const GRID_SIZE: Point2d<u8> = Point2d::splat(0);

    const GRID_OVERLAP: u8 = 1;

    const SIZE: Point2d<u8> = Point2d::splat(0);

    fn compute(
        (layer,): &<Self::Dependencies as Dependencies>::Layer,
        index: GridPoint<Self>,
    ) -> Self {
        for _ in layer.get_range(Self::bounds(index)) {}
        Player
    }
}

#[derive(Clone, Default)]
struct MapChunk;

impl Chunk for MapChunk {
    type LayerStore<T> = T;
    type Dependencies = (TheChunk,);

    const SIZE: Point2d<u8> = Point2d::splat(0);

    const GRID_SIZE: Point2d<u8> = Point2d::splat(0);

    const GRID_OVERLAP: u8 = 1;

    fn compute(
        (layer,): &<Self::Dependencies as Dependencies>::Layer,
        index: GridPoint<Self>,
    ) -> Self {
        for _ in layer.get_range(Self::bounds(index)) {}
        MapChunk
    }
}

#[test]
fn create_layer() {
    let layer = Layer::new(());
    layer.get_or_compute(Point2d { x: 42, y: 99 }.map(GridIndex::<TheChunk>::from_raw));
}

#[test]
fn double_assign_chunk() {
    let layer = Layer::new(());
    layer.get_or_compute(Point2d { x: 42, y: 99 }.map(GridIndex::<TheChunk>::from_raw));
    // This is very incorrect, but adding assertions for checking its
    // correctness destroys all caching and makes logging and perf
    // completely useless.
    layer.get_or_compute(Point2d { x: 42, y: 99 }.map(GridIndex::<TheChunk>::from_raw));
}

#[test]
fn create_player() {
    init_tracing();
    let the_layer = Layer::new(());
    let player = Layer::<Player>::new((the_layer.clone(),));
    let player_pos = Point2d { x: 42, y: 99 };
    player.ensure_loaded_in_bounds(Bounds::point(player_pos));
    let map = Layer::<MapChunk>::new((the_layer,));
    map.ensure_loaded_in_bounds(Bounds::point(player_pos));
}
