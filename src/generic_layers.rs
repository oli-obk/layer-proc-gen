//! Various useful layer/chunk type combinations that you can reuse in many kind of games.

use arrayvec::ArrayVec;
use rand::prelude::*;

use crate::{
    Chunk,
    debug::{Debug, DebugContent},
    rolling_grid::GridPoint,
    vec2::{Num, Point2d},
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

#[derive(PartialEq, Debug, Clone)]
/// A type of chunk that contains on average one point.
/// You can specify a size in real world coordinates as well as
/// a random number generator salt for picking different points
/// even for the same chunk coordinates.
pub struct UniformPoint<P, const SIZE: u8, const SALT: u64> {
    /// The actual points. Can be up to 7, as a poisson distribution of one point
    /// per chunk has a negligible probability for more than 7 points.
    pub points: ArrayVec<P, 7>,
}

impl<P, const SIZE: u8, const SALT: u64> Default for UniformPoint<P, SIZE, SALT> {
    fn default() -> Self {
        Self {
            points: Default::default(),
        }
    }
}

impl<P: Reducible, const SIZE: u8, const SALT: u64> Chunk for UniformPoint<P, SIZE, SALT> {
    type LayerStore<T> = T;
    type Dependencies = ();

    const SIZE: Point2d<u8> = Point2d::splat(SIZE);

    fn compute((): &Self::Dependencies, index: GridPoint<Self>) -> Self {
        let points = generate_points::<SALT, Self>(index);
        Self {
            points: points.map(P::from).collect(),
        }
    }

    fn clear((): &Self::Dependencies, _index: GridPoint<Self>) {
        // Nothing to do, we do not have dependencies
    }
}

impl<P: Reducible, const SIZE: u8, const SALT: u64> Debug for UniformPoint<P, SIZE, SALT> {
    fn debug(&self) -> Vec<DebugContent> {
        self.points.iter().flat_map(Reducible::debug).collect()
    }
}

fn generate_points<const SALT: u64, C: Chunk + 'static>(
    index: GridPoint<C>,
) -> impl Iterator<Item = Point2d> {
    let chunk_bounds = C::bounds(index);
    let mut rng = rng_for_point::<SALT, _>(index);
    let n = poisson_1(rng.random_range(0.0..=1.0)).into();
    std::iter::from_fn(move || Some(chunk_bounds.sample(&mut rng))).take(n)
}

/// Create a random number generator seeded with a specific point.
pub fn rng_for_point<const SALT: u64, T: Num>(index: Point2d<T>) -> SmallRng {
    let x = SmallRng::seed_from_u64(index.x.as_u64());
    let y = SmallRng::seed_from_u64(index.y.as_u64());
    let salt = SmallRng::seed_from_u64(SALT);
    let mut seed = <SmallRng as SeedableRng>::Seed::default();
    for mut rng in [x, y, salt] {
        let mut tmp = <SmallRng as SeedableRng>::Seed::default();
        rng.fill_bytes(&mut tmp);
        for (seed, tmp) in seed.iter_mut().zip(tmp.iter()) {
            *seed ^= *tmp;
        }
    }
    SmallRng::from_seed(seed)
}

mod reduced_points;
pub use reduced_points::*;
