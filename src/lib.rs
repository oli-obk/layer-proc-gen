//! A Rust implementation of https://github.com/runevision/LayerProcGen
//!
//! You implement pairs of layers and chunks, e.g. ExampleLayer and ExampleChunk. A layer contains chunks of the corresponding type.

#![warn(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
#![allow(async_fn_in_trait)]

use std::{
    cell::{Ref, RefCell},
    collections::HashMap,
    future::Future,
    num::NonZeroU16,
    sync::Arc,
};

use rolling_grid::{GridIndex, GridPoint, RollingGrid};
use scheduler::Scheduler;
use tracing::{debug_span, instrument, trace};
use vec2::{Bounds, Point2d};

pub mod scheduler;

/// Each layer stores a RollingGrid of corresponding chunks.
pub trait Layer: Sized {
    /// Corresponding `Chunk` type. A `Layer` type must always be paired with exactly one `Chunk` type.
    type Chunk: Chunk<Layer = Self>;

    /// Internal `RollingGrid` size.
    const GRID_SIZE: Point2d<u8> = Point2d::splat(32);
    /// Internal `RollingGrid` overlap before the system panics. Basically scales the grid width/height by
    /// this number to allow moving across the grid width/height boundaries completely transparently.
    /// Increasing this number makes indexing the `RollingGrid` more expensive if there is a lot of overlap.
    const GRID_OVERLAP: u8 = 3;

    fn rolling_grid(&self) -> &RollingGrid<Self>;

    /// Returns the chunk that the position is in and the position within the chunk
    fn get_chunk_of_grid_point(&self, pos: Point2d) -> Option<(Ref<'_, Self::Chunk>, Point2d)> {
        let chunk_pos = RollingGrid::<Self>::pos_to_grid_pos(pos);
        let chunk = self.rolling_grid().get(chunk_pos)?;
        Some((chunk, RollingGrid::<Self>::pos_within_chunk(pos, chunk_pos)))
    }

    /// Load all dependencies' chunks and then compute our chunks.
    /// May recursively cause the dependencies to load their deps and so on.
    #[instrument(level = "trace", skip(self), fields(this = std::any::type_name::<Self>()))]
    async fn ensure_loaded_in_bounds(&self, bounds: Bounds<i64>) {
        let indices = Self::Chunk::bounds_to_grid(bounds);
        trace!(?indices);
        let mut create_indices: Vec<_> = indices.iter().collect();
        let center = indices.center();
        // Sort by distance to center, so we load the closest ones first
        // Difference to
        create_indices.sort_by_cached_key(|&index| index.dist_squared(center));
        for index in create_indices {
            self.create_and_register_chunk(index).await;
        }
    }

    /// Load a single chunk.
    #[instrument(level = "trace", skip(self), fields(this = std::any::type_name::<Self>()))]
    async fn create_and_register_chunk(&self, index: GridPoint) {
        self.rolling_grid()
            .set(index, || async move {
                let span = debug_span!("compute", ?index, layer = std::any::type_name::<Self>());
                let _guard = span.enter();
                self.ensure_chunk_providers(index).await;
                Self::Chunk::compute(self, index).await
            })
            .await
    }

    /// Load a single chunks' dependencies.
    #[instrument(level = "trace", skip(self), fields(this = std::any::type_name::<Self>()))]
    async fn ensure_chunk_providers(&self, index: GridPoint) {
        let chunk_bounds = Self::Chunk::bounds(index);
        self.ensure_all_deps(chunk_bounds).await;
    }

    /// Invoke `ensure_loaded_in_bounds` on all your dependencies here.
    async fn ensure_all_deps(&self, chunk_bounds: Bounds);
}

/// Actual way to access dependency layers. Handles generating and fetching the right blocks.
// FIXME: use `const PADDING: Point2d`
/// The Padding is in game coordinates.
pub struct LayerDependency<L: Layer, const PADDING_X: i64, const PADDING_Y: i64> {
    layer: Arc<L>,
    scheduler: Scheduler,
    scheduled_nodes: RefCell<HashMap<GridPoint, Arc<dyn Future<Output = ()>>>>,
}

impl<L: Layer, const PADDING_X: i64, const PADDING_Y: i64>
    LayerDependency<L, PADDING_X, PADDING_Y>
{
    pub const fn padding(&self) -> Point2d {
        Point2d::new(PADDING_X, PADDING_Y)
    }

    /// Eagerly load all chunks in the given bounds (in world coordinates).
    pub async fn ensure_loaded_in_bounds(&self, chunk_bounds: Bounds) {
        let required_bounds = chunk_bounds.pad(self.padding());
        self.layer.ensure_loaded_in_bounds(required_bounds).await;
    }

    /// Get a chunk or panic if it was not loaded previously
    pub async fn get(&self, index: GridPoint) -> Ref<'_, L::Chunk> {
        match self.layer.rolling_grid().get(index) {
            Some(chunk) => chunk,
            None => {
                let guard = self
                    .scheduled_nodes
                    .borrow_mut()
                    .entry(index)
                    .or_insert_with(|| {
                        let layer = self.layer.clone();
                        Arc::new(async move { layer.create_and_register_chunk(index).await })
                    })
                    .clone();
                guard.await;
                self.layer.rolling_grid().get(index).unwrap()
            }
        }
    }

    /// Get an iterator over all chunks that touch the given bounds (in world coordinates)
    pub fn get_range(
        &self,
        range: Bounds,
    ) -> impl Iterator<Item = impl Future<Output = Ref<'_, L::Chunk>>> {
        let range = L::Chunk::bounds_to_grid(range);
        self.get_grid_range(range)
    }

    /// Get an iterator over chunks as given by the bounds (in chunk grid indices)
    pub fn get_grid_range(
        &self,
        range: Bounds<GridIndex>,
    ) -> impl Iterator<Item = impl Future<Output = Ref<'_, L::Chunk>>> {
        range.iter().map(move |pos| self.get(pos))
    }
}

impl<L: Layer, const PADDING_X: i64, const PADDING_Y: i64> From<Arc<L>>
    for LayerDependency<L, PADDING_X, PADDING_Y>
{
    fn from(layer: Arc<L>) -> Self {
        Self { layer }
    }
}

/// Chunks are always rectangular and all chunks in a given layer have the same world space size.
pub trait Chunk: Sized {
    /// Corresponding `Layer` type. A `Chunk` type must always be paired with exactly one `Layer` type.
    type Layer: Layer<Chunk = Self>;
    /// Width and height of the chunk
    const SIZE: Point2d<NonZeroU16> = match NonZeroU16::new(256) {
        Some(v) => Point2d::splat(v),
        None => unreachable!(),
    };

    /// Compute a chunk from its dependencies
    async fn compute(layer: &Self::Layer, index: GridPoint) -> Self;

    /// Get the bounds for the chunk at the given index
    fn bounds(index: GridPoint) -> Bounds {
        let min = index.map(|GridIndex(i)| i) * Point2d::from(Self::SIZE);
        Bounds {
            min,
            max: min + Self::SIZE.into(),
        }
    }

    /// Get the grids that are touched by the given bounds.
    fn bounds_to_grid(bounds: Bounds) -> Bounds<GridIndex> {
        bounds.map(Self::pos_to_grid)
    }

    /// Get the grid the position is in
    fn pos_to_grid(point: Point2d) -> GridPoint {
        point.div_euclid(Self::SIZE).map(GridIndex)
    }
}

pub mod rolling_grid;
pub mod vec2;
