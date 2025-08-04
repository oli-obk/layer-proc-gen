use std::sync::Arc;

use arrayvec::ArrayVec;

use crate::{
    Chunk, ChunkExt as _, Dependencies, Layer,
    debug::{Debug, DebugContent},
    rolling_grid::GridPoint,
    vec2::{Bounds, Point2d},
};

use super::UniformPoint;

/// Represents point like types that do not want to be close to other types.
/// The highest [priority][`Reducible::priority`] (largest radius, by default) of two objects is kept if they are too close to each other.
/// If both objects have the same priority, the one with the higher X coordinate is kept (or higher Y if X is also the same).
pub trait Reducible: PartialEq + Clone + Sized + 'static {
    /// The type that will be passed into [`Reducible::try_new`] as context when creating an instance of this type.
    type Dependencies: Dependencies;

    /// Attempt to create an instance of this type at the given point.
    /// If [`None`] is returned, the point will be skipped.
    fn try_new(center: Point2d, deps: &Self::Dependencies) -> Option<Self>;
    /// The maximum radius that things of this type can be, with the given context.
    ///
    /// Used to scan for overlap, OK to overestimate but the larger it is the more chunks need to be scanned.
    fn max_radius(deps: &Self::Dependencies) -> i64;
    /// The radius around the thing to be kept free from other things.
    fn radius(&self) -> i64;
    /// Center position of the circle to keep free of other things.
    fn position(&self) -> Point2d;
    /// The priority of the thing, used to determine the "winner" when there's overlap.
    fn priority(&self) -> i64 {
        self.radius()
    }
    /// Debug representation. Usually contains just a single thing, the item itself,
    /// but can be overriden to emit addition information.
    fn debug(&self, _bounds: Bounds) -> Vec<DebugContent> {
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
    type Dependencies = Layer<UniformPoint<P, SIZE, SALT>>;
    const SIZE: Point2d<u8> = Point2d::splat(SIZE);

    fn compute(raw_points: &Self::Dependencies, index: GridPoint<Self>) -> Self {
        let max_radius = P::max_radius(&raw_points.1);
        let mut points = ArrayVec::new();
        'points: for p in raw_points.get(index.into_same_chunk_size()).points {
            for other in raw_points
                .get_range(Bounds::point(p.position()).pad(Point2d::splat(p.radius() + max_radius)))
            {
                for other in other.points {
                    if other == p {
                        continue;
                    }

                    // prefer to delete lower priority, then lower x, then lower y
                    let lower_priority = p
                        .priority()
                        .cmp(&other.priority())
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

    fn clear(raw_points: &Self::Dependencies, index: GridPoint<Self>) {
        raw_points.clear(Self::bounds(index));
    }
}

impl<P: Reducible, const SIZE: u8, const SALT: u64> Debug for ReducedUniformPoint<P, SIZE, SALT> {
    fn debug(&self, bounds: Bounds) -> Vec<DebugContent> {
        self.points
            .iter()
            .flat_map(|p| {
                let mut debug = p.debug(bounds);
                for debug in &mut debug {
                    // After reducing, the radius is irrelevant and it is nicer to represent it as a point.
                    match debug {
                        DebugContent::Chunk => {}
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
