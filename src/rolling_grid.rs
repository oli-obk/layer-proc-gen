use crate::{vec2::Point2d, Chunk, Layer};

pub struct RollingGrid<L: Layer> {
    grid: Box<[Cell<L>]>,
}

impl<L: Layer> Default for RollingGrid<L> {
    fn default() -> Self {
        Self {
            grid: std::iter::repeat_with(Cell::default)
                .take(usize::from(L::GRID_SIZE.x) * usize::from(L::GRID_SIZE.y))
                .collect(),
        }
    }
}

/// Contains up to `L::OVERLAP` entries
#[expect(clippy::type_complexity)]
struct Cell<L: Layer>(Box<[Option<(Point2d, L::Chunk)>]>);

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
            .find(|(p, _)| *p == point)
            .map(|(_, chunk)| chunk)
    }

    fn set(&mut self, point: Point2d, chunk: L::Chunk) {
        for (p, _) in self.0.iter().flatten() {
            assert_ne!(*p, point);
        }
        let data = self
            .0
            .iter_mut()
            .find(|o| o.is_none())
            .expect("overlap exceeded");
        *data = Some((point, chunk));
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

    pub fn get(&self, point: Point2d) -> Option<&L::Chunk> {
        self.grid[Self::index_of_point(point)].get(point)
    }

    pub fn set(&mut self, point: Point2d, chunk: L::Chunk) {
        self.grid[Self::index_of_point(point)].set(point, chunk)
    }
}
