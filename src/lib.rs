//! A Rust implementation of https://github.com/runevision/LayerProcGen
//!
//! You implement pairs of layers and chunks, e.g. ExampleLayer and ExampleChunk. A layer contains chunks of the corresponding type.

#![warn(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]

use std::{cell::Ref, num::NonZeroU16, sync::Arc};

use rolling_grid::{GridIndex, GridPoint, RollingGrid};
use tracing::{debug_span, instrument, trace};
use vec2::{GridBounds, Point2d};

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
    #[track_caller]
    #[instrument(level = "trace", skip(self), fields(this = std::any::type_name::<Self>()))]
    fn ensure_loaded_in_bounds(&self, bounds: GridBounds<i64>) {
        let indices = Self::Chunk::bounds_to_grid(bounds);
        trace!(?indices);
        let mut create_indices: Vec<_> = indices.iter().collect();
        let center = indices.center();
        // Sort by distance to center, so we load the closest ones first
        // Difference to
        create_indices.sort_by_cached_key(|&index| index.dist_squared(center));
        for index in create_indices {
            self.create_and_register_chunk(index);
        }
    }

    /// Load a single chunk.
    #[track_caller]
    #[instrument(level = "trace", skip(self), fields(this = std::any::type_name::<Self>()))]
    fn create_and_register_chunk(&self, index: GridPoint) {
        self.rolling_grid().set(index, || {
            let span = debug_span!("compute", ?index, layer = std::any::type_name::<Self>());
            let _guard = span.enter();
            self.ensure_chunk_providers(index);
            Self::Chunk::compute(self, index)
        })
    }

    /// Load a single chunks' dependencies.
    #[instrument(level = "trace", skip(self), fields(this = std::any::type_name::<Self>()))]
    fn ensure_chunk_providers(&self, index: GridPoint) {
        let chunk_bounds = Self::Chunk::bounds(index);
        self.ensure_all_deps(chunk_bounds);
    }

    /// Invoke `ensure_loaded_in_bounds` on all your dependencies here.
    fn ensure_all_deps(&self, chunk_bounds: GridBounds);
}

/// Actual way to access dependency layers. Handles generating and fetching the right blocks.
// FIXME: use `const PADDING: Point2d`
/// The Padding is in game coordinates.
pub struct LayerDependency<L: Layer, const PADDING_X: i64, const PADDING_Y: i64> {
    layer: Arc<L>,
}

impl<L: Layer, const PADDING_X: i64, const PADDING_Y: i64>
    LayerDependency<L, PADDING_X, PADDING_Y>
{
    pub fn ensure_loaded_in_bounds(&self, chunk_bounds: GridBounds) {
        let required_bounds = chunk_bounds.pad(Point2d::new(PADDING_X, PADDING_Y));
        self.layer.ensure_loaded_in_bounds(required_bounds);
    }

    /// Get a chunk or panic if it was not loaded previously
    pub fn get(&self, index: GridPoint) -> Ref<'_, L::Chunk> {
        self.layer.rolling_grid().get(index).unwrap_or_else(|| {
            panic!(
                "chunk at {index:?} is not yet loaded in {}",
                std::any::type_name::<L>()
            )
        })
    }

    pub fn get_range(&self, range: GridBounds) -> impl Iterator<Item = Ref<'_, L::Chunk>> {
        let range = L::Chunk::bounds_to_grid(range);
        self.layer
            .rolling_grid()
            .get_range(range)
            .map(move |chunk| {
                chunk.unwrap_or_else(|| {
                    panic!(
                        "a chunk in {range:?} is not yet loaded in {}",
                        std::any::type_name::<L>()
                    )
                })
            })
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
    fn compute(layer: &Self::Layer, index: GridPoint) -> Self;

    /// Get the bounds for the chunk at the given index
    fn bounds(index: GridPoint) -> GridBounds {
        let min = index.map(|GridIndex(i)| i) * Self::SIZE.into();
        GridBounds {
            min,
            max: min + Self::SIZE.into(),
        }
    }

    /// Get the grids that are touched by the given bounds.
    fn bounds_to_grid(bounds: GridBounds) -> GridBounds<GridIndex> {
        bounds.map(Self::pos_to_grid)
    }

    /// Get the grid the position is in
    fn pos_to_grid(point: Point2d) -> GridPoint {
        (point / Point2d::from(Self::SIZE)).map(GridIndex)
    }
}

pub mod rolling_grid;
pub mod vec2;
