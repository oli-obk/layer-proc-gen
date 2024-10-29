use crate::{
    vec2::{Abs, Num, Point2d},
    Chunk, Layer,
};
use std::{
    cell::{Cell, RefCell},
    marker::PhantomData,
    ops::{Div, DivAssign, Neg},
    time::{SystemTime, UNIX_EPOCH},
};

pub type GridPoint<C> = crate::vec2::Point2d<GridIndex<C>>;

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
            .take((1 << L::GRID_SIZE.x) << L::GRID_SIZE.y)
            .collect(),
        }
    }
}

pub struct GridIndex<C>(pub i64, PhantomData<C>);

impl<C> Abs for GridIndex<C> {
    fn abs(self) -> Self {
        Self::from_raw(self.0.abs())
    }
}

impl<C> Div for GridIndex<C> {
    type Output = Self;

    fn div(mut self, rhs: Self) -> Self::Output {
        self /= rhs;
        self
    }
}

impl<C> DivAssign for GridIndex<C> {
    fn div_assign(&mut self, rhs: Self) {
        self.0 /= rhs.0;
    }
}

impl<C> std::ops::SubAssign for GridIndex<C> {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
    }
}

impl<C> std::ops::Sub for GridIndex<C> {
    type Output = Self;

    fn sub(mut self, rhs: Self) -> Self::Output {
        self -= rhs;
        self
    }
}

impl<C> std::ops::Mul for GridIndex<C> {
    type Output = Self;

    fn mul(mut self, rhs: Self) -> Self::Output {
        self *= rhs;
        self
    }
}

impl<C> std::ops::MulAssign for GridIndex<C> {
    fn mul_assign(&mut self, rhs: Self) {
        self.0 *= rhs.0;
    }
}

impl<C> std::ops::Add for GridIndex<C> {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl<C> std::ops::AddAssign for GridIndex<C> {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl<C> Ord for GridIndex<C> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}

impl<C> PartialOrd for GridIndex<C> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.0.cmp(&other.0))
    }
}

impl<C> Eq for GridIndex<C> {}

impl<C> PartialEq for GridIndex<C> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<C> std::fmt::Debug for GridIndex<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("GridIndex").field(&self.0).finish()
    }
}

impl<C> Copy for GridIndex<C> {}

impl<C> Clone for GridIndex<C> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<C> GridIndex<C> {
    pub const fn from_raw(i: i64) -> Self {
        Self(i, PhantomData)
    }
}

impl<C> Neg for GridIndex<C> {
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        self.0 = self.0.neg();
        self
    }
}

impl<C> Num for GridIndex<C> {
    const ZERO: Self = Self::from_raw(0);
    const ONE: Self = Self::from_raw(1);
    const TWO: Self = Self::from_raw(2);

    fn iter_range(mut range: std::ops::Range<Self>) -> impl Iterator<Item = Self> {
        std::iter::from_fn(move || {
            if range.start == range.end {
                None
            } else {
                let i = range.start;
                range.start.0 += 1;
                Some(i)
            }
        })
    }

    fn as_u64(self) -> u64 {
        self.0.as_u64()
    }
}

impl<C> Div<i64> for GridIndex<C> {
    type Output = Self;
    fn div(mut self, rhs: i64) -> Self::Output {
        self /= rhs;
        self
    }
}

impl<C> DivAssign<i64> for GridIndex<C> {
    fn div_assign(&mut self, rhs: i64) {
        self.0 /= rhs;
    }
}

impl<C: Chunk> GridPoint<C> {
    pub fn into_same_chunk_size<D: Chunk>(self) -> GridPoint<D> {
        const { assert!(C::SIZE.x == D::SIZE.x && C::SIZE.y == D::SIZE.y) };
        GridPoint {
            x: GridIndex::from_raw(self.x.0),
            y: GridIndex::from_raw(self.y.0),
        }
    }
}

struct ActiveCell<L: Layer> {
    pos: Cell<GridPoint<L::Chunk>>,
    chunk: RefCell<<L::Chunk as Chunk>::Store>,
    last_access: Cell<SystemTime>,
}

impl<L: Layer> Default for ActiveCell<L> {
    fn default() -> Self {
        Self {
            pos: GridPoint::splat(GridIndex::from_raw(i64::MIN)).into(),
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
    pub fn get_or_compute(
        &self,
        pos: GridPoint<L::Chunk>,
        layer: &L,
    ) -> <L::Chunk as Chunk>::Store {
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

    pub const fn pos_within_chunk(pos: Point2d, chunk_pos: GridPoint<L::Chunk>) -> Point2d {
        pos.sub(Point2d {
            x: chunk_pos.x.0 * (1 << L::Chunk::SIZE.x),
            y: chunk_pos.y.0 * (1 << L::Chunk::SIZE.y),
        })
    }

    pub const fn pos_to_grid_pos(pos: Point2d) -> GridPoint<L::Chunk> {
        GridPoint {
            x: GridIndex::from_raw(pos.x >> L::Chunk::SIZE.x),
            y: GridIndex::from_raw(pos.y >> L::Chunk::SIZE.y),
        }
    }

    const fn index_of_point(point: GridPoint<L::Chunk>) -> usize {
        const { assert!((L::GRID_SIZE.x as u32) < usize::BITS) }
        const { assert!((L::GRID_SIZE.y as u32) < usize::BITS) }
        #[expect(
            clippy::cast_possible_truncation,
            reason = "checked above that remainder op will alway fit in usize"
        )]
        let x = point.x.0.rem_euclid(1 << L::GRID_SIZE.x) as usize;
        #[expect(
            clippy::cast_possible_truncation,
            reason = "checked above that remainder op will alway fit in usize"
        )]
        let y = point.y.0.rem_euclid(1 << L::GRID_SIZE.y) as usize;
        x + (y << L::GRID_SIZE.x)
    }

    #[track_caller]
    fn access(&self, pos: GridPoint<L::Chunk>) -> &[ActiveCell<L>] {
        self.grid
            .get(Self::index_of_point(pos))
            .unwrap_or_else(|| panic!("grid position {pos:?} out of bounds"))
    }
}
