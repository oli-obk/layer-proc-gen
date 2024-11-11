use crate::{
    vec2::{Abs, Num, Point2d},
    Chunk, Dependencies,
};
use std::{
    cell::{Cell, RefCell},
    marker::PhantomData,
    ops::{Div, DivAssign, Neg},
};

/// The x and y positions of a chunk in the number of chunks, not in world coordinates.
pub type GridPoint<C> = crate::vec2::Point2d<GridIndex<C>>;

// TODO: avoid the box when generic const exprs allow for it
// The Layer that contains it will already get put into an `Arc`
pub(crate) struct RollingGrid<C: Chunk> {
    /// The inner slice contains to `L::OVERLAP` entries,
    /// some of which are `None` if they have nevef been used
    /// so far.
    grid: Box<[Box<[ActiveCell<C>]>]>,
    time: Cell<u64>,
}

impl<C: Chunk> Default for RollingGrid<C> {
    fn default() -> Self {
        Self {
            grid: std::iter::repeat_with(|| {
                std::iter::repeat_with(Default::default)
                    .take(C::GRID_OVERLAP.into())
                    .collect()
            })
            .take((1 << C::GRID_SIZE.x) << C::GRID_SIZE.y)
            .collect(),
            time: Cell::new(1),
        }
    }
}

/// An x or y index in chunk coordinates, not world coordinates.
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
    /// Create a [GridIndex] for any chunk.
    /// Useful if you are doing some custom math to convert from a
    /// [Chunk]'s coordinates to another [Chunk]'s coordinates without
    /// going through world coordinates.
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
    /// When two [Chunk]s have the same size, all their coordinates are trivially
    /// the same and we can convert them with just a compile-time check.
    pub fn into_same_chunk_size<D: Chunk>(self) -> GridPoint<D> {
        const { assert!(C::SIZE.x == D::SIZE.x && C::SIZE.y == D::SIZE.y) };
        GridPoint {
            x: GridIndex::from_raw(self.x.0),
            y: GridIndex::from_raw(self.y.0),
        }
    }
}

struct ActiveCell<C: Chunk> {
    pos: Cell<GridPoint<C>>,
    chunk: RefCell<C>,
    last_access: Cell<u64>,
}

impl<C: Chunk> Default for ActiveCell<C> {
    fn default() -> Self {
        Self {
            pos: GridPoint::splat(GridIndex::from_raw(i64::MIN)).into(),
            chunk: Default::default(),
            last_access: Cell::new(0),
        }
    }
}

impl<C: Chunk> RollingGrid<C> {
    #[track_caller]
    /// If the position is already occupied with a block,
    /// debug assert that it's the same that we'd generate.
    /// Otherwise just increment the user count for that block.
    pub fn get_or_compute(
        &self,
        pos: GridPoint<C>,
        layer: &<C::Dependencies as Dependencies>::Layer,
    ) -> C {
        let now = self.time.get();
        self.time.set(now.checked_add(1).unwrap());
        // Find existing entry and bump its last use, or
        // find an empty entry, or
        // find the least recently accessed entry.
        let (mut free, mut rest) = self.access(pos).split_first().unwrap();
        while let Some((p, r)) = rest.split_first() {
            rest = r;
            if p.last_access.get() == 0 {
            } else if p.pos.get() == pos {
                p.last_access.set(now);
                return p.chunk.borrow().clone();
            } else if free.last_access < p.last_access {
                continue;
            }
            free = p;
        }
        let chunk = C::compute(layer, pos);
        free.pos.set(pos);
        free.chunk.replace(chunk.clone());
        free.last_access.set(now);
        chunk
    }

    pub const fn pos_to_grid_pos(pos: Point2d) -> GridPoint<C> {
        GridPoint {
            x: GridIndex::from_raw(pos.x >> C::SIZE.x),
            y: GridIndex::from_raw(pos.y >> C::SIZE.y),
        }
    }

    const fn index_of_point(point: GridPoint<C>) -> usize {
        const { assert!((C::GRID_SIZE.x as u32) < usize::BITS) }
        const { assert!((C::GRID_SIZE.y as u32) < usize::BITS) }
        #[expect(
            clippy::cast_possible_truncation,
            reason = "checked above that remainder op will alway fit in usize"
        )]
        let x = point.x.0.rem_euclid(1 << C::GRID_SIZE.x) as usize;
        #[expect(
            clippy::cast_possible_truncation,
            reason = "checked above that remainder op will alway fit in usize"
        )]
        let y = point.y.0.rem_euclid(1 << C::GRID_SIZE.y) as usize;
        x + (y << C::GRID_SIZE.x)
    }

    #[track_caller]
    fn access(&self, pos: GridPoint<C>) -> &[ActiveCell<C>] {
        self.grid
            .get(Self::index_of_point(pos))
            .unwrap_or_else(|| panic!("grid position {pos:?} out of bounds"))
    }

    pub fn iter_all_loaded(&self) -> impl Iterator<Item = (GridPoint<C>, C)> + '_ {
        self.grid
            .iter()
            .flatten()
            .filter(|cell| cell.last_access.get() != 0)
            .map(|cell| (cell.pos.get(), cell.chunk.borrow().clone()))
    }
}
