use std::sync::Arc;

use arrayvec::ArrayVec;

use crate::{
    rolling_grid::GridPoint,
    vec2::{Bounds, Point2d},
    Chunk, Layer, LayerDependency,
};

use super::UniformPointLayer;

pub trait Reducible: From<Point2d> + PartialEq + Clone + Sized + 'static {
    /// The radius around the thing to be kept free from other things.
    fn radius(&self) -> i64;
    fn position(&self) -> Point2d;
}

/// Removes locations that are too close to others
pub struct ReducedUniformPointLayer<P: Reducible, const SIZE: u8, const SALT: u64> {
    points: LayerDependency<UniformPointLayer<P, SIZE, SALT>>,
}

impl<P: Reducible, const SIZE: u8, const SALT: u64> Default
    for ReducedUniformPointLayer<P, SIZE, SALT>
{
    fn default() -> Self {
        Self {
            points: Default::default(),
        }
    }
}

#[derive(PartialEq, Debug, Clone)]
pub struct ReducedUniformPointChunk<P, const SIZE: u8, const SALT: u64> {
    pub points: ArrayVec<P, 7>,
}

impl<P, const SIZE: u8, const SALT: u64> Default for ReducedUniformPointChunk<P, SIZE, SALT> {
    fn default() -> Self {
        Self {
            points: Default::default(),
        }
    }
}

impl<P: Reducible, const SIZE: u8, const SALT: u64> Layer
    for ReducedUniformPointLayer<P, SIZE, SALT>
{
    type Chunk = ReducedUniformPointChunk<P, SIZE, SALT>;
    type Store<T> = Arc<T>;

    fn ensure_all_deps(&self, chunk_bounds: Bounds) {
        self.points.ensure_loaded_in_bounds(chunk_bounds);
    }
}

impl<P: Reducible, const SIZE: u8, const SALT: u64> Chunk
    for ReducedUniformPointChunk<P, SIZE, SALT>
{
    type Layer = ReducedUniformPointLayer<P, SIZE, SALT>;
    type Store = Self;
    const SIZE: Point2d<u8> = Point2d::splat(SIZE);

    fn compute(layer: &Self::Layer, index: GridPoint<Self>) -> Self {
        let mut points = ArrayVec::new();
        'points: for p in layer
            .points
            .get_or_compute(index.into_same_chunk_size())
            .points
        {
            for other in layer.points.get_range(Bounds {
                min: p.position(),
                max: p.position() + Point2d::splat(p.radius()),
            }) {
                for other in other.points {
                    if other == p {
                        continue;
                    }

                    if other.position().manhattan_dist(p.position()) < p.radius() + other.radius()
                        && p.radius() < other.radius()
                    {
                        continue 'points;
                    }
                }
            }
            points.push(p);
        }
        ReducedUniformPointChunk { points }
    }
}
