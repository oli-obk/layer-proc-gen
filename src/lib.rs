//! A Rust implementation of https://github.com/runevision/LayerProcGen
//!
//! You implement pairs of layers and chunks, e.g. ExampleLayer and ExampleChunk. A layer contains chunks of the corresponding type.

#![warn(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]

use std::{num::NonZeroU16, sync::Arc};

use rolling_grid::RollingGrid;
use vec2::Point2d;

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
    fn rolling_grid_mut(&mut self) -> &mut RollingGrid<Self>;

    /// Returns the chunk that the position is in and the position within the chunk
    fn get_chunk_of_grid_point(&self, pos: Point2d) -> Option<(&Self::Chunk, Point2d)> {
        let chunk_pos = RollingGrid::<Self>::pos_to_grid_pos(pos);
        let chunk = self.rolling_grid().get(chunk_pos)?;
        Some((chunk, RollingGrid::<Self>::pos_within_chunk(chunk_pos, pos)))
    }
}

/// Actual way to access dependency layers. Handles generating and fetching the right blocks.
// FIXME: use `const PADDING: Point2d`
/// The Padding is in game coordinates.
pub struct LayerDependency<L: Layer, const PADDING_X: i64, const PADDING_Y: i64> {
    layer: Arc<L>,
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
}

pub mod rolling_grid;
pub mod vec2;
