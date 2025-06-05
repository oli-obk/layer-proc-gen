use debug::{Debug, DebugContent};
use macroquad::prelude::*;
use std::ops::Deref;

use layer_proc_gen::{
    generic_layers::{ContourLines, Density, LookupIndex},
    vec2::Num as _,
    *,
};

#[derive(Debug, Clone, Default)]
struct Block {
    density: bool,
}

impl Chunk for Block {
    type LayerStore<T> = T;
    const SIZE: Point2d<u8> = Point2d::splat(5);

    fn compute((): &Self::Dependencies, index: GridPoint<Self>) -> Self {
        Self {
            density: index.len_squared().0.isqrt() < 10,
        }
    }

    fn clear((): &Self::Dependencies, _index: GridPoint<Self>) {
        todo!()
    }

    type Dependencies = ();
}

impl Debug for Block {
    fn debug(&self, _bounds: Bounds) -> Vec<DebugContent> {
        if self.density {
            vec![DebugContent::Chunk]
        } else {
            vec![]
        }
    }
}

impl Density for Block {
    fn above_threshold(&self) -> bool {
        self.density
    }
}

#[macroquad::main("layer proc gen demo")]
async fn main() {
    let lines = Layer::<ContourLines<Block>>::default();

    loop {
        if is_key_pressed(KeyCode::Escape) {
            return;
        }

        let range = Bounds {
            min: Point2d::splat(GridIndex::ZERO),
            max: Point2d {
                x: GridIndex::from_raw(15),
                y: GridIndex::from_raw(15),
            },
        };

        for pos in range.iter() {
            let bounds = Block::bounds(pos).map(|d| d.map(|d| d as f32));

            let block = lines.deref().deref().get(pos);
            if block.density {
                draw_rectangle(
                    bounds.min.x,
                    bounds.min.y,
                    bounds.max.x - bounds.min.x,
                    bounds.max.y - bounds.min.y,
                    GRAY,
                );
            }
            let chunk = lines.deref().get(pos.into_same_chunk_size());
            draw_text(
                &chunk.idx.to_string(),
                bounds.min.x,
                bounds.min.y,
                20.,
                WHITE,
            );
            let chunk = lines.get(pos.into_same_chunk_size());
            for shape in chunk.debug(Block::bounds(pos)) {
                match shape {
                    DebugContent::Chunk => draw_rectangle(
                        bounds.min.x,
                        bounds.min.y,
                        bounds.max.x - bounds.min.x,
                        bounds.max.y - bounds.min.y,
                        GRAY,
                    ),
                    DebugContent::Line(line) => {
                        let line = line.map(|p| p.map(|p| p as f32));
                        draw_line(line.start.x, line.start.y, line.end.x, line.end.y, 1., RED)
                    }
                    DebugContent::Circle { .. } => todo!(),
                    DebugContent::Text { .. } => todo!(),
                }
            }
        }

        next_frame().await
    }
}
