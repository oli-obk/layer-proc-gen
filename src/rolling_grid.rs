use crate::{
    vec2::{Num, Point2d},
    Chunk, Layer,
};
use derive_more::derive::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};
use std::cell::{Ref, RefCell};
use std::ops::{Div, DivAssign};

pub type GridPoint = crate::vec2::Point2d<GridIndex>;

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

#[derive(
    Copy,
    Clone,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    AddAssign,
    Add,
    Mul,
    MulAssign,
    Sub,
    SubAssign,
    Div,
    DivAssign,
    Debug,
)]
#[mul(forward)]
#[div(forward)]
#[mul_assign(forward)]
#[div_assign(forward)]
pub struct GridIndex(pub i64);

impl Num for GridIndex {
    const ONE: GridIndex = GridIndex(1);
    const TWO: GridIndex = GridIndex(2);
}

impl Div<i64> for GridIndex {
    type Output = Self;
    fn div(mut self, rhs: i64) -> Self::Output {
        self /= rhs;
        self
    }
}

impl DivAssign<i64> for GridIndex {
    fn div_assign(&mut self, rhs: i64) {
        self.0 /= rhs;
    }
}

/// Contains up to `L::OVERLAP` entries
struct Cell<L: Layer>(Box<[Option<ActiveCell<L>>]>);

struct ActiveCell<L: Layer> {
    pos: GridPoint,
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
    fn get(&self, point: GridPoint) -> Option<&L::Chunk> {
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
    fn set(&mut self, pos: GridPoint, chunk: impl FnOnce() -> L::Chunk) {
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
    pub const fn pos_within_chunk(pos: Point2d, chunk_pos: GridPoint) -> Point2d {
        pos.sub(
            Point2d {
                x: chunk_pos.x.0,
                y: chunk_pos.y.0,
            }
            .mul(L::Chunk::SIZE),
        )
    }

    pub const fn pos_to_grid_pos(pos: Point2d) -> GridPoint {
        let pos = pos.div_euclid(L::Chunk::SIZE);
        GridPoint {
            x: GridIndex(pos.x),
            y: GridIndex(pos.y),
        }
    }

    const fn index_of_point(point: GridPoint) -> usize {
        let point = Point2d {
            x: point.x.0,
            y: point.y.0,
        }
        .rem_euclid(L::GRID_SIZE);
        point.x + point.y * L::GRID_SIZE.x as usize
    }

    pub fn get(&self, pos: GridPoint) -> Option<Ref<'_, L::Chunk>> {
        Ref::filter_map(self.access(pos).borrow(), |cell| cell.get(pos)).ok()
    }

    #[track_caller]
    pub fn set(&self, pos: GridPoint, chunk: impl FnOnce() -> L::Chunk) {
        self.access(pos).borrow_mut().set(pos, chunk)
    }

    #[track_caller]
    fn access(&self, pos: GridPoint) -> &RefCell<Cell<L>> {
        self.grid
            .get(Self::index_of_point(pos))
            .unwrap_or_else(|| panic!("grid position {pos:?} out of bounds"))
    }
}
