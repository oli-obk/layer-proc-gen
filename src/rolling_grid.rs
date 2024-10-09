use crate::{vec2::Point2d, Chunk, Layer};

pub struct RollingGrid<L: Layer> {
    grid: Box<[Cell<L>]>,
}

impl<L: Layer> RollingGrid<L> {
    #[expect(
        clippy::cast_possible_wrap,
        reason = "checked with compile time assert _SMALL_WIDTH"
    )]
    const WIDTH: i64 = L::GRID_WIDTH as i64;
    #[expect(
        clippy::cast_possible_wrap,
        reason = "checked with compile time assert _SMALL_HEIGHT"
    )]
    const HEIGHT: i64 = L::GRID_HEIGHT as i64;
    const _POSITIVE_WIDTH: () = { assert!(Self::WIDTH > 0) };
    const _POSITIVE_HEIGHT: () = { assert!(Self::HEIGHT > 0) };
    const _SMALL_WIDTH: () = { assert!(Self::WIDTH < i16::MAX as i64) };
    const _SMALL_HEIGHT: () = { assert!(Self::HEIGHT < i16::MAX as i64) };
    const _NO_8_BIT_USIZE: () = { assert!(usize::MAX > u8::MAX as usize) };
}

/// Contains up to `L::OVERLAP` entries
#[expect(clippy::type_complexity)]
struct Cell<L: Layer>(Box<[Option<(Point2d, L::Chunk)>]>);

impl<L: Layer> Cell<L> {
    fn get(&self, index: Point2d) -> Option<&L::Chunk> {
        self.0
            .iter()
            .filter_map(|e| e.as_ref())
            .find(|(p, _)| *p == index)
            .map(|(_, chunk)| chunk)
    }

    fn set(&mut self, index: Point2d, chunk: L::Chunk) {
        for (p, _) in self.0.iter().flatten() {
            assert_ne!(*p, index);
        }
        let data = self
            .0
            .iter_mut()
            .find(|o| o.is_none())
            .expect("overlap exceeded");
        *data = Some((index, chunk));
    }
}

impl<L: Layer> RollingGrid<L> {
    const fn pos(index: Point2d) -> usize {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "checked with compile time assert _SMALL_WIDTH"
        )]
        let x = index.x.rem_euclid(Self::WIDTH) as usize;
        #[expect(
            clippy::cast_possible_truncation,
            reason = "checked with compile time assert _SMALL_HEIGHT"
        )]
        let y = index.y.rem_euclid(Self::HEIGHT) as usize;
        x + y * L::Chunk::WIDTH.get()
    }

    pub fn get(&self, index: Point2d) -> Option<&L::Chunk> {
        self.grid[Self::pos(index)].get(index)
    }

    pub fn set(&mut self, index: Point2d, chunk: L::Chunk) {
        self.grid[Self::pos(index)].set(index, chunk)
    }
}
