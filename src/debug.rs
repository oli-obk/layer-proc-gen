//! Various helpers for viewing layers and their data without knowing the exact structure and contents

use std::{any::TypeId, borrow::Borrow as _};

use crate::{
    rolling_grid::RollingGrid,
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

    fn deps(&self) -> Vec<&dyn DynLayer>;
    fn ident(&self) -> (usize, TypeId);
    fn name(&self) -> String;
}

impl<C: Chunk> DynLayer for Layer<C> {
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

    fn deps(&self) -> Vec<&dyn DynLayer> {
        self.debug_deps()
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
