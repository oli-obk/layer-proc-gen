//! A Rust implementation of <https://github.com/runevision/LayerProcGen>
//!
//! The main entry point to access the data of layers is the [Layer] type.
//!
//! To create your own layers, you need to define a data type for the [Chunk]s of
//! that layer, and implement the [Chunk] trait for it. This trait also allows you
//! to specify other [Chunk] types whose layers you'll need information from to generate
//! the data of your own chunks.
//!
//! All coordinates are integer based (chunk position and borders), but the data within your
//! chunks can be of arbitrary data types, including floats, strings, etc.
//!
//! As an example, here's a layer that generates a point at its center:
//!
//! ```rust
//! use layer_proc_gen::{Chunk, GridPoint, Point2d, debug::DynLayer, ChunkExt as _};
//!
//! #[derive(Clone, Default)]
//! struct MyChunk {
//!     center: Point2d,
//! }
//!
//! impl Chunk for MyChunk {
//!     type LayerStore<T> = std::sync::Arc<T>;
//!     type Dependencies = ();
//!     fn compute(&(): &(), index: GridPoint<Self>) -> Self {
//!         let center = Self::bounds(index).center();
//!         MyChunk { center }
//!     }
//!     fn clear(&(): &(), index: GridPoint<Self>) {
//!         // No dependencies to clear things from
//!     }
//! }
//! ```
//!
//! There are builtin [Chunk] types for common patterns like generating uniformly
//! distributed points on an infinite grid.

#![warn(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
#![deny(missing_docs)]

use std::{borrow::Borrow, ops::Deref};

use debug::DynLayer;
use rolling_grid::RollingGrid;
pub use rolling_grid::{GridIndex, GridPoint};
pub use vec2::{Bounds, Point2d};

pub mod debug;
pub mod generic_layers;

#[macro_export]
/// Generate a struct where all fields are wrapped in `Layer`
macro_rules! deps {
    ($(#[$meta:meta])* struct $name:ident {$($field:ident: $ty:ty,)*}) => {
        $(#[$meta])*
        struct $name {
            $($field: Layer<$ty>,)*
        }
        impl $crate::Dependencies for $name {
            fn debug(&self) -> Vec<&dyn $crate::debug::DynLayer> {
                let $name {
                    $($field,)*
                } = self;
                vec![ $($field,)*]
            }
        }
    }
}

/// A struct that defines the dependencies of your [Chunk].
/// Usually generated for structs via the [deps] macro, but you can manually define
/// it in case you have non-[Layer] dependencies.
///
/// If you layer has no dependencies, you can use the `()` type instead.
pub trait Dependencies {
    /// For runtime debugging of your layers, you should return references to each of the
    /// layer types within your dependencies.
    fn debug(&self) -> Vec<&dyn DynLayer>;
}

impl Dependencies for () {
    fn debug(&self) -> Vec<&dyn DynLayer> {
        vec![]
    }
}

impl<C: Chunk + debug::Debug> Dependencies for Layer<C> {
    fn debug(&self) -> Vec<&dyn DynLayer> {
        vec![self]
    }
}

/// Wrapper around a [`u64`] seed so it can be used as [`Dependencies`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Seed(pub u64);

impl Dependencies for Seed {
    fn debug(&self) -> Vec<&dyn crate::debug::DynLayer> {
        vec![]
    }
}

impl<D: Dependencies> Dependencies for (Seed, D) {
    fn debug(&self) -> Vec<&dyn DynLayer> {
        self.1.debug()
    }
}

/// The entry point to access the chunks of a layer.
///
/// It exposes various convenience accessors, like iterating over areas in
/// chunk or world coordinates.
pub struct Layer<C: Chunk> {
    layer: Store<C>,
}

impl<C: Chunk> Deref for Layer<C> {
    type Target = C::Dependencies;

    fn deref(&self) -> &Self::Target {
        &self.layer.borrow().1
    }
}

impl<C: Chunk> Default for Layer<C>
where
    C::Dependencies: Default,
{
    /// Create an entirely new layer and its dependencies.
    /// The dependencies will not be connected to any other dependencies
    /// of the same type.
    fn default() -> Self {
        Self::new(Default::default())
    }
}

impl<C: Chunk> Layer<C> {
    /// Create a new layer, manually specifying the dependencies.
    /// This is useful if you want to share dependencies with another layer.
    pub fn new(value: C::Dependencies) -> Self {
        Layer {
            layer: Store::<C>::from((RollingGrid::default(), value)),
        }
    }
}

impl<C: Chunk> Drop for Layer<C> {
    fn drop(&mut self) {
        let data = self.layer.borrow();
        data.0.drop(&data.1);
    }
}

impl<C: Chunk> Clone for Layer<C>
where
    Store<C>: Clone,
{
    /// Shallowly clone the layer. The clones will share the caches with this
    /// copy of the layer.
    fn clone(&self) -> Self {
        Self {
            layer: self.layer.clone(),
        }
    }
}

#[expect(type_alias_bounds)]
type Store<C: Chunk> = C::LayerStore<Tuple<C>>;
#[expect(type_alias_bounds)]
type Tuple<C: Chunk> = (RollingGrid<C>, C::Dependencies);

impl<C: Chunk> Layer<C> {
    /// Eagerly compute all chunks in the given bounds (in world coordinates).
    /// Load all dependencies' chunks and then compute our chunks.
    /// May recursively cause the dependencies to load their deps and so on.
    #[track_caller]
    pub fn ensure_loaded_in_bounds(&self, chunk_bounds: Bounds) {
        let indices = C::bounds_to_grid(chunk_bounds);
        let mut create_indices: Vec<_> = indices.iter().collect();
        let center = indices.center();
        // Sort by distance to center, so we load the closest ones first
        // Difference to
        create_indices.sort_by_cached_key(|&index| index.dist_squared(center));
        for index in create_indices {
            self.get(index);
        }
    }

    /// Eagerly unload all chunks in the given bounds (in world coordinates).
    pub fn clear(&self, chunk_bounds: Bounds) {
        for index in C::bounds_to_grid(chunk_bounds).iter() {
            self.layer.borrow().0.clear(index, self)
        }
    }

    /// Manually (without calling `compute`) set a chunk in the cache.
    ///
    /// This violates all the nice properties like the fact that layers
    /// depending on this one will not even load this chunk if they have
    /// computed all their chunks that depend on this one. They may
    /// later have to recompute, and then see the new value, which may
    /// lead to recomputed chunks being different from non-recomputed chunks.
    ///
    /// TLDR: only call this if you have called `clear` on everything that depended
    /// on this one.
    pub fn incoherent_override_cache(&self, index: GridPoint<C>, val: C) {
        self.layer
            .borrow()
            .0
            .incoherent_override_cache(self, index, val)
    }

    /// Get a chunk or generate it if it wasn't already cached.
    pub fn get(&self, index: GridPoint<C>) -> C {
        self.layer.borrow().0.get(index, self)
    }

    /// Get an iterator over all chunks that touch the given bounds (in world coordinates)
    pub fn get_range(&self, range: Bounds) -> impl Iterator<Item = C> + '_ {
        let range = C::bounds_to_grid(range);
        self.get_grid_range(range)
    }

    /// Get an iterator over chunks as given by the bounds (in chunk grid indices).
    /// Chunks will be generated on the fly.
    pub fn get_grid_range(&self, range: Bounds<GridIndex<C>>) -> impl Iterator<Item = C> + '_ {
        // TODO: first request generation, then iterate to increase parallelism
        range.iter().map(move |pos| self.get(pos))
    }

    /// Get a 3x3 array of chunks around a specific chunk
    pub fn get_moore_neighborhood(&self, index: GridPoint<C>) -> [[C; 3]; 3] {
        C::moore_neighborhood(index).map(|line| line.map(|index| self.get(index)))
    }
}

/// Chunks are rectangular regions of the same size that make up a layer in a grid shape.
pub trait Chunk: Sized + Default + Clone + 'static {
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

    /// Width and height of the chunk (in powers of two);
    const SIZE: Point2d<u8> = Point2d::splat(8);

    /// Compute a chunk from its dependencies
    fn compute(layer: &Self::Dependencies, index: GridPoint<Self>) -> Self;

    /// Clear all information that [compute] would have computed
    fn clear(layer: &Self::Dependencies, index: GridPoint<Self>);

    /// Called on [Drop], [Chunk::clear] or when
    /// the chunk gets overwritten because it was too old.
    fn on_drop(&self, _layer: &Self::Dependencies, _index: GridPoint<Self>) {}

    /// The actual dependencies. Usually a struct with fields of `Layer<T>` type, but
    /// can be of any type to specify non-layer dependencies, too.
    /// It is the type of the first argument of [Chunk::compute].
    type Dependencies: Dependencies;
}

/// Various helpers that a [Chunk] frequently needs
pub trait ChunkExt: Chunk {
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

    /// Pad by a chunk size to make sure we see effects from the neighboring chunks
    fn vision_range(bounds: Bounds) -> Bounds {
        bounds.pad(Self::SIZE.map(|i| 1 << i))
    }

    /// Get 3x3 grid points around a central one
    fn moore_neighborhood(index: GridPoint<Self>) -> [[GridPoint<Self>; 3]; 3] {
        let p = |x, y| index + GridPoint::new(GridIndex::from_raw(x), GridIndex::from_raw(y));
        [
            [p(-1, -1), p(0, -1), p(1, -1)],
            [p(-1, 0), p(0, 0), p(1, 0)],
            [p(-1, 1), p(0, 1), p(1, 1)],
        ]
    }
}

impl<T: Chunk> ChunkExt for T {}

mod rolling_grid;
pub mod vec2;
