//! Various useful layer/chunk type combinations that you can reuse in many kind of games.

use std::marker::PhantomData;

use arrayvec::ArrayVec;
use rand::prelude::*;

use crate::{
    rolling_grid::GridPoint,
    vec2::{Bounds, Num, Point2d},
    Chunk, Layer,
};

/// How many points are in a chunk if the
/// average is 1 point. Input is a value picked
/// from a uniform distribution in 0..1
fn poisson_1(val: f32) -> u8 {
    match val {
        0.0..0.3679 => 0,
        0.3670..0.7358 => 1,
        0.7358..0.9197 => 2,
        0.9197..0.981 => 3,
        0.981..0.9963 => 4,
        0.9963..0.9994 => 5,
        0.9994..0.9999 => 6,
        0.9999..1.0 => 7,
        _ => panic!("{val} is not in range 0..1"),
    }
}

pub struct UniformPointLayer<P, const SIZE: u8, const SALT: u64>(PhantomData<P>);

impl<P, const SIZE: u8, const SALT: u64> Default for UniformPointLayer<P, SIZE, SALT> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

#[derive(PartialEq, Debug, Clone)]
pub struct UniformPointChunk<P, const SIZE: u8, const SALT: u64> {
    pub points: ArrayVec<P, 7>,
}

impl<P, const SIZE: u8, const SALT: u64> Default for UniformPointChunk<P, SIZE, SALT> {
    fn default() -> Self {
        Self {
            points: Default::default(),
        }
    }
}

impl<P: From<Point2d> + Clone + 'static, const SIZE: u8, const SALT: u64> Layer
    for UniformPointLayer<P, SIZE, SALT>
{
    type Chunk = UniformPointChunk<P, SIZE, SALT>;

    fn ensure_all_deps(&self, _chunk_bounds: Bounds) {}
}

impl<P: From<Point2d> + Clone + 'static, const SIZE: u8, const SALT: u64> Chunk
    for UniformPointChunk<P, SIZE, SALT>
{
    type LayerStore<T> = T;
    type Layer = UniformPointLayer<P, SIZE, SALT>;
    type Store = Self;

    const SIZE: Point2d<u8> = Point2d::splat(SIZE);

    fn compute(_layer: &Self::Layer, index: GridPoint<Self>) -> Self {
        let points = generate_points::<SALT, Self>(index);
        Self {
            points: points.map(P::from).collect(),
        }
    }
}

fn generate_points<const SALT: u64, C: Chunk + 'static>(
    index: GridPoint<C>,
) -> impl Iterator<Item = Point2d> {
    let chunk_bounds = C::bounds(index);
    let mut rng = rng_for_point::<SALT, _>(index);
    let n = poisson_1(rng.gen_range(0.0..=1.0)).into();
    std::iter::from_fn(move || Some(chunk_bounds.sample(&mut rng))).take(n)
}

/// Create a random number generator seeded with a specific point.
pub fn rng_for_point<const SALT: u64, T: Num>(index: Point2d<T>) -> SmallRng {
    let x = SmallRng::seed_from_u64(index.x.as_u64());
    let y = SmallRng::seed_from_u64(index.y.as_u64());
    let salt = SmallRng::seed_from_u64(SALT);
    let mut seed = [0; 32];
    for mut rng in [x, y, salt] {
        let mut tmp = [0; 32];
        rng.fill_bytes(&mut tmp);
        for (seed, tmp) in seed.iter_mut().zip(tmp.iter()) {
            *seed ^= *tmp;
        }
    }
    SmallRng::from_seed(seed)
}

mod reduced_points;
pub use reduced_points::*;
