use std::sync::Arc;

use layer_proc_gen::*;
use vec2::{Bounds, Point2d};

#[expect(dead_code)]
#[derive(Clone, Default)]
struct TheChunk(usize);

impl Chunk for TheChunk {
    type LayerStore<T> = Arc<T>;
    type Dependencies = ();

    fn compute(_layer: &Self::Dependencies, _index: GridPoint<Self>) -> Self {
        TheChunk(0)
    }

    fn clear(_layer: &Self::Dependencies, _index: GridPoint<Self>) {}
}

impl debug::Debug for TheChunk {}

#[derive(Clone, Default)]
struct Player;

deps! {
    #[derive(Clone, Default)]
    struct PlayerDeps {
        layer: TheChunk,
    }
}

impl Chunk for Player {
    type LayerStore<T> = T;
    type Dependencies = PlayerDeps;

    const GRID_SIZE: Point2d<u8> = Point2d::splat(0);

    const GRID_OVERLAP: u8 = 1;

    const SIZE: Point2d<u8> = Point2d::splat(0);

    fn compute(deps: &Self::Dependencies, index: GridPoint<Self>) -> Self {
        for _ in deps.layer.get_range(Self::bounds(index)) {}
        Player
    }

    fn clear(_layer: &Self::Dependencies, _index: GridPoint<Self>) {}
}

#[derive(Clone, Default)]
struct MapChunk;

impl Chunk for MapChunk {
    type LayerStore<T> = T;
    type Dependencies = PlayerDeps;

    const SIZE: Point2d<u8> = Point2d::splat(0);

    const GRID_SIZE: Point2d<u8> = Point2d::splat(0);

    const GRID_OVERLAP: u8 = 1;

    fn compute(deps: &Self::Dependencies, index: GridPoint<Self>) -> Self {
        for _ in deps.layer.get_range(Self::bounds(index)) {}
        MapChunk
    }

    fn clear(_layer: &Self::Dependencies, _index: GridPoint<Self>) {}
}

#[test]
fn create_layer() {
    let layer = Layer::<TheChunk>::new(());
    layer.get(Point2d { x: 42, y: 99 }.map(GridIndex::from_raw));
}

#[test]
fn double_assign_chunk() {
    let layer = Layer::<TheChunk>::new(());
    layer.get(Point2d { x: 42, y: 99 }.map(GridIndex::from_raw));
    // This is very incorrect, but adding assertions for checking its
    // correctness destroys all caching and makes logging and perf
    // completely useless.
    layer.get(Point2d { x: 42, y: 99 }.map(GridIndex::from_raw));
}

#[test]
fn create_player() {
    let the_layer = PlayerDeps::default();
    let player = Layer::<Player>::new(the_layer.clone());
    let player_pos = Point2d { x: 42, y: 99 };
    player.ensure_loaded_in_bounds(Bounds::point(player_pos));
    let map = Layer::<MapChunk>::new(the_layer.clone());
    map.ensure_loaded_in_bounds(Bounds::point(player_pos));
}
