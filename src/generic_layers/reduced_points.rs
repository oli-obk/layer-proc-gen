use std::sync::Arc;

use arrayvec::ArrayVec;

use crate::{
    debug::DebugContent,
    rolling_grid::GridPoint,
    vec2::{Bounds, Point2d},
    Chunk, Dependencies,
};

use super::UniformPoint;

pub trait Reducible: From<Point2d> + PartialEq + Clone + Sized + 'static {
    /// The radius around the thing to be kept free from other things.
    fn radius(&self) -> i64;
    fn position(&self) -> Point2d;
    fn debug(&self) -> Vec<DebugContent> {
        vec![DebugContent::Circle {
            center: self.position(),
            radius: self.radius() as f32,
        }]
    }
}

#[derive(PartialEq, Debug, Clone)]
/// Removes locations that are too close to others.
pub struct ReducedUniformPoint<P, const SIZE: u8, const SALT: u64> {
    pub points: ArrayVec<P, 7>,
}

impl<P, const SIZE: u8, const SALT: u64> Default for ReducedUniformPoint<P, SIZE, SALT> {
    fn default() -> Self {
        Self {
            points: Default::default(),
        }
    }
}

impl<P: Reducible, const SIZE: u8, const SALT: u64> Chunk for ReducedUniformPoint<P, SIZE, SALT> {
    type LayerStore<T> = Arc<T>;
    type Dependencies = (UniformPoint<P, SIZE, SALT>,);
    const SIZE: Point2d<u8> = Point2d::splat(SIZE);

    fn compute(
        (raw_points,): &<Self::Dependencies as Dependencies>::Layer,
        index: GridPoint<Self>,
    ) -> Self {
        let mut points = ArrayVec::new();
        'points: for p in raw_points
            .get_or_compute(index.into_same_chunk_size())
            .points
        {
            for other in raw_points.get_range(Bounds {
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
        ReducedUniformPoint { points }
    }

    fn debug_contents(&self) -> Vec<DebugContent> {
        self.points
            .iter()
            .flat_map(|p| {
                let mut debug = p.debug();
                for debug in &mut debug {
                    // After reducing, the radius is irrelevant and it is nicer to represent it as a point.
                    match debug {
                        DebugContent::Line(..) => {}
                        DebugContent::Circle { radius, .. } => *radius = 1.,
                        DebugContent::Text { .. } => {}
                    }
                }
                debug
            })
            .collect()
    }
}
