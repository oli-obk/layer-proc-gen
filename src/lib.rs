//! A Rust implementation of https://github.com/runevision/LayerProcGen
//!
//! You implement pairs of layers and chunks, e.g. ExampleLayer and ExampleChunk. A layer contains chunks of the corresponding type.

#![warn(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]

use std::borrow::Borrow;

use rolling_grid::{GridIndex, GridPoint, RollingGrid};
use tracing::{instrument, trace};
use vec2::{Bounds, Point2d};

pub mod generic_layers;

/// Each layer stores a RollingGrid of corresponding chunks.
pub trait Dependencies {
    type AsLayerDependencies: Default;
}

macro_rules! layer {
    ($($t:ident,)*) => {
        impl<$($t: Chunk,)*> Dependencies
            for ($($t,)*)
        {
            type AsLayerDependencies = ($(Layer<$t>,)*);
        }
    };
}
macro_rules! layers {
    ($($first:ident,)* =>) => {
        layer!($($first,)*);
    };
    ($($first:ident,)* => $next:ident, $($t:ident,)*) => {
        layer!($($first,)*);
        layers!($($first,)* $next, => $($t,)*);
    };
}

layers!(=> T, U, V,);

/// Actual way to access dependency layers. Handles generating and fetching the right blocks.
pub struct Layer<C: Chunk> {
    layer: Store<C>,
}

impl<C: Chunk> Default for Layer<C> {
    fn default() -> Self {
        Self {
            layer: Store::<C>::from(Default::default()),
        }
    }
}

impl<C: Chunk> Layer<C> {
    pub fn new(value: <C::Dependencies as Dependencies>::AsLayerDependencies) -> Self {
        Layer {
            layer: Store::<C>::from((RollingGrid::default(), value)),
        }
    }
}

impl<C: Chunk> Clone for Layer<C>
where
    Store<C>: Clone,
{
    fn clone(&self) -> Self {
        Self {
            layer: self.layer.clone(),
        }
    }
}

#[expect(type_alias_bounds)]
type Store<C: Chunk> = C::LayerStore<Tuple<C>>;
#[expect(type_alias_bounds)]
type Tuple<C: Chunk> = (
    RollingGrid<C>,
    <C::Dependencies as Dependencies>::AsLayerDependencies,
);

impl<C: Chunk> Layer<C> {
    /// Eagerly compute all chunks in the given bounds (in world coordinates).
    /// Load all dependencies' chunks and then compute our chunks.
    /// May recursively cause the dependencies to load their deps and so on.
    #[track_caller]
    #[instrument(level = "trace", skip(self), fields(this = std::any::type_name::<C>()))]
    pub fn ensure_loaded_in_bounds(&self, chunk_bounds: Bounds) {
        let indices = C::bounds_to_grid(chunk_bounds);
        trace!(?indices);
        let mut create_indices: Vec<_> = indices.iter().collect();
        let center = indices.center();
        // Sort by distance to center, so we load the closest ones first
        // Difference to
        create_indices.sort_by_cached_key(|&index| index.dist_squared(center));
        for index in create_indices {
            self.get_or_compute(index);
        }
    }

    /// Get a chunk or generate it if it wasn't already cached.
    pub fn get_or_compute(&self, index: GridPoint<C>) -> C::Store {
        self.layer
            .borrow()
            .0
            .get_or_compute(index, &self.layer.borrow().1)
    }

    /// Get an iterator over all chunks that touch the given bounds (in world coordinates)
    pub fn get_range(&self, range: Bounds) -> impl Iterator<Item = C::Store> + '_ {
        let range = C::bounds_to_grid(range);
        self.get_grid_range(range)
    }

    /// Get an iterator over chunks as given by the bounds (in chunk grid indices).
    /// Chunks will be generated on the fly.
    pub fn get_grid_range(
        &self,
        range: Bounds<GridIndex<C>>,
    ) -> impl Iterator<Item = C::Store> + '_ {
        // TODO: first request generation, then iterate to increase parallelism
        range.iter().map(move |pos| self.get_or_compute(pos))
    }
}

/// Chunks are always rectangular and all chunks in a given layer have the same world space size.
pub trait Chunk: Sized + 'static {
    /// Exponent of `2` of the cached area (in grid chunk numbers, not world coordinates).
    /// This is the area that should stay in memory at all times as it will get requested a lot.
    const GRID_SIZE: Point2d<u8> = Point2d::splat(5);
    /// Internal `RollingGrid` overlap before the system drops old chunks. Basically scales the grid width/height by
    /// this number to allow moving across the grid width/height boundaries completely transparently.
    /// Increasing this number makes indexing the `RollingGrid` more expensive if there is a lot of overlap.
    const GRID_OVERLAP: u8 = 3;

    /// Data structure that stores the layer. Usually `Arc<Self>`,
    /// but some layers are only used to simplify another layer, so
    /// they can get stored directly without the `Arc` indirection.
    type LayerStore<T>: Borrow<T> + From<T>;

    /// Tuple of `LayerDependency` that `compute` needs access to.
    type Dependencies: Dependencies;

    /// For small and cheap to clone `Chunk` types, just use `Self` for `Store`,
    /// otherwise any thread safe shared smart pointer type will suffice, usually `Arc<Self>`.
    type Store: Clone + Borrow<Self> + Default;

    /// Width and height of the chunk (in powers of two);
    const SIZE: Point2d<u8> = Point2d::splat(8);

    /// Compute a chunk from its dependencies
    fn compute(
        layer: &<Self::Dependencies as Dependencies>::AsLayerDependencies,
        index: GridPoint<Self>,
    ) -> Self::Store;

    /// Get the bounds for the chunk at the given index
    fn bounds(index: GridPoint<Self>) -> Bounds {
        let size = Self::SIZE.map(|i| 1 << i);
        let min = index.map(|i| i.0) * size;
        Bounds {
            min,
            max: min + size,
        }
    }

    /// Get the grids that are touched by the given bounds.
    fn bounds_to_grid(bounds: Bounds) -> Bounds<GridIndex<Self>> {
        bounds.map(Self::pos_to_grid)
    }

    /// Get the grid the position is in
    fn pos_to_grid(point: Point2d) -> GridPoint<Self> {
        RollingGrid::<Self>::pos_to_grid_pos(point)
    }

    fn default_layer() -> <Self::Dependencies as Dependencies>::AsLayerDependencies {
        Default::default()
    }
}

pub mod rolling_grid;
pub mod vec2;
