use std::{ops::Range, sync::Arc};

use arrayvec::ArrayVec;

use crate::{
    debug::DebugContent,
    rolling_grid::GridPoint,
    vec2::{Bounds, Point2d},
    Chunk, Dependencies,
};

use super::UniformPoint;

/// Represents point like types that do not want to be close to other types.
/// The larger of two objects is kept if they are too close to each other.
/// If both objects have the same radius, the one with the higher X coordinate is kept (or higher Y if X is also the same).
pub trait Reducible: From<Point2d> + PartialEq + Clone + Sized + 'static {
    /// The range of radii that `radius` can return.
    const RADIUS_RANGE: Range<i64>;

    /// The radius around the thing to be kept free from other things.
    fn radius(&self) -> i64;
    /// Center position of the circle to keep free of other things.
    fn position(&self) -> Point2d;
    /// Debug representation. Usually contains just a single thing, the item itself,
    /// but can be overriden to emit addition information.
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
    /// The points remaining after removing ones that are too close to others.
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
            for other in raw_points.get_range(
                Bounds::point(p.position()).pad(Point2d::splat(p.radius() + P::RADIUS_RANGE.end)),
            ) {
                for other in other.points {
                    if other == p {
                        continue;
                    }

                    // prefer to delete lower radius, then lower x, then lower y
                    let lower_priority = p
                        .radius()
                        .cmp(&other.radius())
                        .then_with(|| p.position().cmp(&other.position()))
                        .is_lt();

                    // skip current point if another point's center is within our radius and we have lower priority
                    if other.position().manhattan_dist(p.position()) < p.radius() + other.radius()
                        && lower_priority
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
