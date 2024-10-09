//! A Rust implementation of https://github.com/runevision/LayerProcGen
//!
//! You implement pairs of layers and chunks, e.g. ExampleLayer and ExampleChunk. A layer contains chunks of the corresponding type.

#![warn(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]

use std::num::NonZeroUsize;

use rolling_grid::RollingGrid;

/// Each layer stores a RollingGrid of corresponding chunks.
pub trait Layer: Sized {
    /// Corresponding `Chunk` type. A `Layer` type must always be paired with exactly one `Chunk` type.
    type Chunk: Chunk;

    /// Internal `RollingGrid` size.
    const GRID_HEIGHT: usize = 32;
    /// Internal `RollingGrid` size.
    const GRID_WIDTH: usize = 32;
    /// Internal `RollingGrid` overlap before the system panics. Basically scales the grid width/height by
    /// this number to allow moving across the grid width/height boundaries completely transparently.
    /// Increasing this number makes indexing the `RollingGrid` more expensive if there is a lot of overlap.
    const GRID_OVERLAP: usize = 3;

    fn rolling_grid(&self) -> &RollingGrid<Self>;
    fn rolling_grid_mut(&mut self) -> &mut RollingGrid<Self>;
}

/// Chunks are always rectangular and all chunks in a given layer have the same world space size.
pub trait Chunk: Sized {
    /// Corresponding `Layer` type. A `Chunk` type must always be paired with exactly one `Layer` type.
    type Layer: Layer;
    /// Width of the chunk
    const WIDTH: NonZeroUsize;
    /// Height of the chunk
    const HEIGHT: NonZeroUsize;
}

mod rolling_grid;
mod vec2;
