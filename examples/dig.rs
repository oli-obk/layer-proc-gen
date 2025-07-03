use debug::{Debug, DebugContent};
use macroquad::prelude::*;
use std::ops::Deref;

use layer_proc_gen::{
    generic_layers::{ContourLines, Density},
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

    fn clear((): &Self::Dependencies, _index: GridPoint<Self>) {}

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

        let (x, y) = mouse_position();
        let index = Block::pos_to_grid(Point2d::new(x as i64, y as i64));

        if is_mouse_button_pressed(MouseButton::Left) | is_mouse_button_pressed(MouseButton::Right)
        {
            let chunk_bounds = Block::bounds(index).pad(Point2d::splat(1));
            lines.clear(chunk_bounds);
            (**lines).incoherent_override_cache(
                index,
                Block {
                    density: is_mouse_button_pressed(MouseButton::Left),
                },
            );
        }

        let range = Bounds {
            min: Point2d::splat(GridIndex::ZERO),
            max: Point2d {
                x: GridIndex::from_raw(15),
                y: GridIndex::from_raw(15),
            },
        };

        if is_key_down(KeyCode::Space) {
            for pos in range.iter() {
                let bounds = Block::bounds(pos).map(|d| d.map(|d| d as f32));

                let block = lines.deref().deref().get(pos);
                if block.density {
                    draw_rectangle(
                        bounds.min.x,
                        bounds.min.y,
                        bounds.max.x - bounds.min.x,
                        bounds.max.y - bounds.min.y,
                        if pos == index { RED } else { GRAY },
                    );
                }
            }
        }

        if is_key_down(KeyCode::Key1) {
            for pos in range.iter() {
                let bounds = Block::bounds(pos).map(|d| d.map(|d| d as f32));

                let chunk = lines.deref().get(pos.into_same_chunk_size());
                draw_text(
                    &chunk.idx.to_string(),
                    bounds.min.x,
                    bounds.min.y,
                    20.,
                    WHITE,
                );
            }
        }

        for pos in range.iter() {
            let chunk = lines.get(pos.into_same_chunk_size());
            for shapes in &chunk.shapes {
                let mesh = Mesh {
                    vertices: shapes
                        .iter()
                        .map(|p| Vertex {
                            position: Vec3::new(p.x as f32, p.y as f32, 0.0),
                            uv: Vec2::new(p.x as f32, p.y as f32),
                            color: [255, 0, 0, 255],
                            normal: Vec4::new(0.0, 0.0, 1.0, 0.0),
                        })
                        .collect(),
                    indices: (0..shapes.len() as u16).chain([0]).collect(),
                    texture: None,
                };
                draw_mesh(&mesh);
            }
        }

        next_frame().await
    }
}
