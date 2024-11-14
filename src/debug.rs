//! Various helpers for viewing layers and their data without knowing the exact structure and contents

use std::{any::TypeId, borrow::Borrow as _};

use crate::{
    rolling_grid::RollingGrid,
    vec2::{Bounds, Line, Point2d},
    Chunk, Dependencies as _, Layer,
};

/// Runtime representation of any chunk type.
/// It is perfectly valid to not return anything if the
/// data is not useful for debug views.
pub trait Debug {
    /// Render the elements of this chunk type.
    fn debug(&self) -> Vec<DebugContent> {
        vec![]
    }
}

/// An debug element of a chunk
pub enum DebugContent {
    /// A line.
    Line(Line),
    /// A unfilled circle.
    Circle {
        /// Center position of the circle
        center: Point2d,
        /// Radius of the circle
        radius: f32,
    },
    /// A Label.
    Text {
        /// Left bottom position of the first line of the text.
        pos: Point2d,
        /// Actual message of the text (can have newlines).
        label: String,
    },
}

impl From<Line> for DebugContent {
    fn from(line: Line) -> Self {
        Self::Line(line)
    }
}

/// Can point to any layer and allows programatic access to dependencies and chunks.
/// Implemented for [Layer]. You should implement this if you manually implement [Dependencies](super::Dependencies).
pub trait DynLayer {
    /// Iterate only over the chunks that have been generated already and not unloaded yet
    /// to make space for new ones.
    fn iter_all_loaded(&self) -> Box<dyn Iterator<Item = (Bounds, Box<dyn Debug + 'static>)> + '_>;

    /// Iterate over the dependency layers of this layer.
    fn deps(&self) -> Vec<&dyn DynLayer>;

    /// A unique identifier for this layer, useful for the use as map keys.
    fn ident(&self) -> (usize, TypeId);

    /// A shortened version of the type name of the layer and its generic parameters.
    fn name(&self) -> String;
}

impl<C: Chunk + Debug> DynLayer for Layer<C> {
    fn iter_all_loaded(&self) -> Box<dyn Iterator<Item = (Bounds, Box<dyn Debug + 'static>)> + '_> {
        Box::new(
            self.layer
                .borrow()
                .0
                .iter_all_loaded()
                .map(|(index, chunk)| {
                    (
                        C::bounds(index),
                        Box::new(chunk) as Box<dyn Debug + 'static>,
                    )
                }),
        )
    }

    fn deps(&self) -> Vec<&dyn DynLayer> {
        self.debug()
    }

    fn ident(&self) -> (usize, TypeId) {
        let ptr: *const RollingGrid<C> = &self.layer.borrow().0;
        (ptr as usize, TypeId::of::<Self>())
    }

    fn name(&self) -> String {
        let mut name = std::any::type_name::<C>().to_owned();
        let mut start = 0;
        loop {
            while let Some((pos, _)) = name[start..]
                .char_indices()
                .take_while(|&(_, c)| matches!(c, 'a'..='z' | 'A'..='Z' | '0'..='9' | '_' | ':'))
                .find(|&(_, c)| c == ':')
            {
                name.replace_range(start..(start + pos + 2), "");
            }
            if let Some((next, c)) = name[start..]
                .char_indices()
                .find(|&(_, c)| !matches!(c,  'a'..='z' | 'A'..='Z' | '0'..='9' | '_'))
            {
                start += next + c.len_utf8();
            } else {
                break;
            }
        }
        name
    }
}
