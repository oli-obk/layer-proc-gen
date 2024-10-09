use crate::{vec2::Point2d, Chunk, Layer};

pub struct RollingGrid<L: Layer> {
    grid: Box<[Cell<L>]>,
}

impl<L: Layer> Default for RollingGrid<L> {
    fn default() -> Self {
        Self {
            grid: std::iter::repeat_with(Cell::default)
                .take(usize::from(L::GRID_WIDTH) * usize::from(L::GRID_HEIGHT))
                .collect(),
        }
    }
}

impl<L: Layer> RollingGrid<L> {
    const _SMALL_CHUNK_WIDTH: () = { assert!(L::Chunk::WIDTH.get() < i16::MAX as usize) };
    const _SMALL_CHUNK_HEIGHT: () = { assert!(L::Chunk::HEIGHT.get() < i16::MAX as usize) };
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
        Point2d {
            #[expect(
                clippy::cast_possible_wrap,
                reason = "checked with compile time assert _SMALL_CHUNK_WIDTH"
            )]
            x: pos.x - chunk_pos.x * L::Chunk::WIDTH.get() as i64,
            #[expect(
                clippy::cast_possible_wrap,
                reason = "checked with compile time assert _SMALL_CHUNK_WIDTH"
            )]
            y: pos.y - chunk_pos.y * L::Chunk::HEIGHT.get() as i64,
        }
    }

    pub const fn pos_to_grid_pos(pos: Point2d) -> Point2d {
        Point2d {
            #[expect(
                clippy::cast_possible_wrap,
                reason = "checked with compile time assert _SMALL_CHUNK_WIDTH"
            )]
            x: pos.x.div_euclid(L::Chunk::WIDTH.get() as i64),
            #[expect(
                clippy::cast_possible_wrap,
                reason = "checked with compile time assert _SMALL_CHUNK_HEIGHT"
            )]
            y: pos.y.div_euclid(L::Chunk::HEIGHT.get() as i64),
        }
    }

    const fn index_of_point(point: Point2d) -> usize {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "remainder op with a u8 will alway fit in usize"
        )]
        let x = point.x.rem_euclid(L::GRID_WIDTH as i64) as usize;
        #[expect(
            clippy::cast_possible_truncation,
            reason = "remainder op with a u8 will alway fit in usize"
        )]
        let y = point.y.rem_euclid(L::GRID_HEIGHT as i64) as usize;
        x + y * L::Chunk::WIDTH.get()
    }

    pub fn get(&self, point: Point2d) -> Option<&L::Chunk> {
        self.grid[Self::index_of_point(point)].get(point)
    }

    pub fn set(&mut self, point: Point2d, chunk: L::Chunk) {
        self.grid[Self::index_of_point(point)].set(point, chunk)
    }
}
