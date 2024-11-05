//! Various helpers for viewing layers and their data without knowing the exact structure and contents

use std::borrow::Borrow as _;

use crate::{
    vec2::{Bounds, Line, Point2d},
    Chunk, Layer,
};

pub trait DynChunk {
    fn render(&self) -> Vec<DebugContent>;
}

impl<C: Chunk> DynChunk for C {
    fn render(&self) -> Vec<DebugContent> {
        self.debug_contents()
    }
}

pub enum DebugContent {
    Line(Line),
    Circle { center: Point2d, radius: f32 },
    Text { pos: Point2d, label: String },
}

impl From<Line> for DebugContent {
    fn from(line: Line) -> Self {
        Self::Line(line)
    }
}

pub trait DynLayer {
    fn iter_all_loaded(
        &self,
    ) -> Box<dyn Iterator<Item = (Bounds, Box<dyn DynChunk + 'static>)> + '_>;
}

impl<C: Chunk> DynLayer for &'_ Layer<C> {
    fn iter_all_loaded(
        &self,
    ) -> Box<dyn Iterator<Item = (Bounds, Box<dyn DynChunk + 'static>)> + '_> {
        Box::new(
            self.layer
                .borrow()
                .0
                .iter_all_loaded()
                .map(|(index, chunk)| {
                    (
                        C::bounds(index),
                        Box::new(chunk) as Box<dyn DynChunk + 'static>,
                    )
                }),
        )
    }
}
