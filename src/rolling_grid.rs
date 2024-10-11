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
    fn set(&mut self, pos: Point2d, chunk: L::Chunk) {
        for p in self.0.iter().flatten() {
            assert_ne!(p.pos, pos);
        }
        match self.0.iter_mut().find(|o| o.is_none()) {
            Some(data) => *data = Some(ActiveCell { pos, chunk }),
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
        point.x + point.y * L::Chunk::SIZE.x.get() as usize
    }

    pub fn get(&self, pos: Point2d) -> Option<Ref<'_, L::Chunk>> {
        Ref::filter_map(self.access(pos).borrow(), |cell| cell.get(pos)).ok()
    }

    #[track_caller]
    pub fn set(&self, pos: Point2d, chunk: L::Chunk) {
        self.access(pos).borrow_mut().set(pos, chunk)
    }

    /// Ensure this chunk does not get removed until we request it
    /// to get removed.
    pub fn increment_user_count(&self, _point: Point2d) {
        todo!()
    }

    fn access(&self, pos: Point2d) -> &RefCell<Cell<L>> {
        &self.grid[Self::index_of_point(pos)]
    }
}
