use crate::{
    vec2::{Num, Point2d},
    Chunk, Layer,
};
use derive_more::derive::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};
use std::{
    cell::{Cell, RefCell},
    ops::{Div, DivAssign},
    time::{SystemTime, UNIX_EPOCH},
};

pub type GridPoint = crate::vec2::Point2d<GridIndex>;

// TODO: avoid the box when generic const exprs allow for it
// The Layer that contains it will already get put into an `Arc`
pub struct RollingGrid<L: Layer> {
    /// The inner slice contains to `L::OVERLAP` entries,
    /// some of which are `None` if they have nevef been used
    /// so far.
    grid: Box<[Box<[ActiveCell<L>]>]>,
}

impl<L: Layer> Default for RollingGrid<L> {
    fn default() -> Self {
        Self {
            grid: std::iter::repeat_with(|| {
                std::iter::repeat_with(Default::default)
                    .take(L::GRID_OVERLAP.into())
                    .collect()
            })
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

struct ActiveCell<L: Layer> {
    pos: Cell<GridPoint>,
    chunk: RefCell<<L::Chunk as Chunk>::Store>,
    last_access: Cell<SystemTime>,
}

impl<L: Layer> Default for ActiveCell<L> {
    fn default() -> Self {
        Self {
            pos: GridPoint::splat(GridIndex(i64::MIN)).into(),
            chunk: Default::default(),
            last_access: UNIX_EPOCH.into(),
        }
    }
}

impl<L: Layer> RollingGrid<L> {
    #[track_caller]
    /// If the position is already occupied with a block,
    /// debug assert that it's the same that we'd generate.
    /// Otherwise just increment the user count for that block.
    pub fn get_or_compute(&self, pos: GridPoint, layer: &L) -> <L::Chunk as Chunk>::Store {
        let now = SystemTime::now();
        // Find existing entry and bump its last use, or
        // find an empty entry, or
        // find the least recently accessed entry.
        let (mut free, mut rest) = self.access(pos).split_first().unwrap();
        while let Some((p, r)) = rest.split_first() {
            rest = r;
            if p.last_access.get() == UNIX_EPOCH {
            } else if p.pos.get() == pos {
                p.last_access.set(now);
                return p.chunk.borrow().clone();
            } else if free.last_access < p.last_access {
                continue;
            }
            free = p;
        }
        let chunk = L::Chunk::compute(layer, pos);
        free.pos.set(pos);
        free.chunk.replace(chunk.clone());
        free.last_access.set(now);
        chunk
    }

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

    #[track_caller]
    fn access(&self, pos: GridPoint) -> &[ActiveCell<L>] {
        self.grid
            .get(Self::index_of_point(pos))
            .unwrap_or_else(|| panic!("grid position {pos:?} out of bounds"))
    }
}
