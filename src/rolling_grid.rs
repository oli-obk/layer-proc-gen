use std::cell::{Ref, RefCell};

use crate::{vec2::Point2d, Chunk, Layer};

pub struct RollingGrid<L: Layer> {
    grid: Box<[RefCell<Cell<L>>]>,
}

impl<L: Layer> Default for RollingGrid<L> {
    fn default() -> Self {
        Self {
            grid: std::iter::repeat_with(Default::default)
                .take(usize::from(L::GRID_SIZE.x) * usize::from(L::GRID_SIZE.y))
                .collect(),
        }
    }
}

/// Contains up to `L::OVERLAP` entries
struct Cell<L: Layer>(Box<[Option<ActiveCell<L>>]>);

struct ActiveCell<L: Layer> {
    pos: Point2d,
    chunk: L::Chunk,
    user_count: usize,
}

impl<L: Layer> Default for Cell<L> {
    fn default() -> Self {
        Self(
            std::iter::repeat_with(|| None)
                .take(L::GRID_OVERLAP.into())
                .collect(),
        )
    }
}

impl<L: Layer> Cell<L> {
    fn get(&self, point: Point2d) -> Option<&L::Chunk> {
        self.0
            .iter()
            .filter_map(|e| e.as_ref())
            .find(|c| c.pos == point)
            .map(|c| &c.chunk)
    }

    #[track_caller]
    /// If the position is already occupied with a block,
    /// debug assert that it's the same that we'd generate.
    /// Otherwise just increment the user count for that block.
    fn set(&mut self, pos: Point2d, chunk: impl FnOnce() -> L::Chunk) {
        let mut free = None;
        for p in self.0.iter_mut() {
            if let Some(p) = p {
                if p.pos == pos {
                    p.user_count += 1;
                    return;
                }
            } else {
                free = Some(p);
            }
        }
        match free {
            Some(data) => {
                *data = Some(ActiveCell {
                    pos,
                    chunk: chunk(),
                    user_count: 0,
                })
            }
            None => {
                let points: Vec<_> = self.0.iter().flatten().map(|c| c.pos).collect();
                panic!("overlap exceeded, could not insert {pos:?}, as we already got {points:?}")
            }
        }
    }
}

impl<L: Layer> RollingGrid<L> {
    pub const fn pos_within_chunk(pos: Point2d, chunk_pos: Point2d) -> Point2d {
        pos.sub(chunk_pos.mul(L::Chunk::SIZE))
    }

    pub const fn pos_to_grid_pos(pos: Point2d) -> Point2d {
        pos.div_euclid(L::Chunk::SIZE)
    }

    const fn index_of_point(point: Point2d) -> usize {
        let point = point.rem_euclid(L::GRID_SIZE);
        point.x + point.y * L::GRID_SIZE.x as usize
    }

    pub fn get(&self, pos: Point2d) -> Option<Ref<'_, L::Chunk>> {
        Ref::filter_map(self.access(pos).borrow(), |cell| cell.get(pos)).ok()
    }

    #[track_caller]
    pub fn set(&self, pos: Point2d, chunk: impl FnOnce() -> L::Chunk) {
        self.access(pos).borrow_mut().set(pos, chunk)
    }

    fn access(&self, pos: Point2d) -> &RefCell<Cell<L>> {
        &self.grid[Self::index_of_point(pos)]
    }
}
