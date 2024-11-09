//! Various position related data structures for 2d integer position handling.

use derive_more::derive::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};
use rand::{
    distributions::{
        uniform::{SampleRange, SampleUniform},
        Standard,
    },
    prelude::*,
};
use std::{
    num::NonZeroU16,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

/// A 2d point where you can choose the type and thus precision of the x and y indices.
/// By default uses [i64] which is the world coordinate type.
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
    Default,
)]
#[mul(forward)]
#[div(forward)]
#[mul_assign(forward)]
#[div_assign(forward)]
pub struct Point2d<T = i64> {
    /// `x` position
    pub x: T,
    /// `y` position
    pub y: T,
}

/// A line segment with a direction.
#[derive(PartialEq, Debug, Copy, Clone)]
pub struct Line<T = i64> {
    /// The start of the line segment.
    pub start: Point2d<T>,
    /// The end of the line segment.
    pub end: Point2d<T>,
}

impl Line {
    /// Returns a point where two line segments intersect (if any).
    // impl from https://stackoverflow.com/a/14795484
    pub fn get_intersection(self, other: Self) -> Option<Point2d> {
        let s10 = self.end - self.start;
        let s32 = other.end - other.start;

        let denom = s10.x * s32.y - s32.x * s10.y;
        if denom == 0 {
            return None; // Collinear
        }
        let denom_positive = denom > 0;

        let s02 = self.start - other.start;
        let s_numer = s10.x * s02.y - s10.y * s02.x;
        if (s_numer < 0) == denom_positive {
            return None; // No collision
        }
        let t_numer = s32.x * s02.y - s32.y * s02.x;
        if (t_numer < 0) == denom_positive {
            return None; // No collision
        }
        if (s_numer > denom) == denom_positive || (t_numer > denom) == denom_positive {
            return None; // No collision
        }
        // Collision detected
        let t = t_numer / denom;

        Some(self.start + s10 * t)
    }

    /// Create bounds where this line is the diagonal of.
    pub fn bounds(&self) -> Bounds {
        Bounds {
            min: self.start,
            max: self.end,
        }
    }

    /// Shorten the line to make its manhattan length the given one.
    pub fn with_manhattan_length(self, len: i64) -> Self {
        assert!(len > 0);
        let dir = self.end - self.start;
        let old_len = dir.x.abs() + dir.y.abs();
        let new_dir = dir * len / old_len;
        Self {
            start: self.start,
            end: self.start + new_dir,
        }
    }

    /// Swap the end and the start.
    pub fn flip(self) -> Self {
        Self {
            start: self.end,
            end: self.start,
        }
    }

    /// Compute the square of the length.
    pub fn len_squared(&self) -> i64 {
        (self.end - self.start).len_squared()
    }
}

impl<T: Num> Line<T> {
    /// Iterate over all pixes that are touched by this line.
    pub fn iter_all_touched_pixels(mut self, mut pnt: impl FnMut(Point2d<T>)) {
        // https://makemeengr.com/precise-subpixel-line-drawing-algorithm-rasterization-algorithm/
        let mut k = Point2d::splat(T::ZERO);
        self.end -= self.start;

        // Pick x direction and step magnitude
        if self.end.x > T::ZERO {
            k.x = T::ONE;
        } else if self.end.x < T::ZERO {
            k.x = -T::ONE;
            self.end.x = -self.end.x;
        }
        self.end.x += T::ONE;

        // Pick y direction and step magnitude
        if self.end.y > T::ZERO {
            k.y = T::ONE;
        } else if self.end.y < T::ZERO {
            k.y = -T::ONE;
            self.end.y = -self.end.y;
        }
        self.end.y += T::ONE;

        // Move in the dimension that steps by more than 1 per step
        let flip = self.end.x >= self.end.y;
        if flip {
            self.end = self.end.flip();
            self.start = self.start.flip();
            k = k.flip();
        }
        let mut pnt = |p: Point2d<T>| if flip { pnt(p.flip()) } else { pnt(p) };

        let mut c = self.end.y;
        for i in T::iter_range(T::ZERO..self.end.y) {
            pnt(self.start); // This is the normal pixel. The two below are subpixels
            c -= self.end.x;
            if c <= T::ZERO {
                if i != self.end.y - T::ONE {
                    pnt(self.start + Point2d::new(T::ZERO, k.y));
                }
                c += self.end.y;
                self.start.x += k.x;
                if i != self.end.y - T::ONE {
                    pnt(self.start);
                }
            }
            self.start.y += k.y
        }
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for Point2d<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        (&self.x, &self.y).fmt(f)
    }
}

impl<T> Distribution<Point2d<T>> for Standard
where
    Standard: Distribution<T>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Point2d<T> {
        Point2d {
            x: self.sample(rng),
            y: self.sample(rng),
        }
    }
}

impl<T: Copy> Point2d<T> {
    /// Set `x` and `y` to the same value
    pub const fn splat(arg: T) -> Self {
        Self::new(arg, arg)
    }

    /// Basic constructor for when struct constructors are too inconvenient
    pub const fn new(x: T, y: T) -> Self {
        Self { x, y }
    }

    /// Apply a closure to both `x` and `y`
    pub fn map<U>(&self, f: impl Fn(T) -> U) -> Point2d<U> {
        Point2d {
            x: f(self.x),
            y: f(self.y),
        }
    }

    /// Connect a line segment from this point to the argument.
    pub fn to(self, other: Self) -> Line<T> {
        Line {
            start: self,
            end: other,
        }
    }

    fn flip(self) -> Point2d<T> {
        Point2d {
            x: self.y,
            y: self.x,
        }
    }
}

impl<T: Neg<Output = T>> Point2d<T> {
    /// Return the perpendicular (right facing) vector of the same length.
    pub fn perp(self) -> Self {
        Self {
            x: -self.y,
            y: self.x,
        }
    }
}

/// Helper trait for computing absolute values of generic types.
pub trait Abs {
    /// Compute the absolute value of this type.
    /// No-op if the value is alread positive.
    fn abs(self) -> Self;
}

impl Abs for i64 {
    fn abs(self) -> Self {
        i64::abs(self)
    }
}

impl<T: Copy + Sub<Output = T> + Mul<Output = T> + Add<Output = T> + Abs> Point2d<T> {
    /// The square of the distance between two points
    pub fn dist_squared(self, center: Point2d<T>) -> T {
        (self - center).len_squared()
    }

    /// The square of the distance between the origin (`0, 0`) and this point.
    pub fn len_squared(self) -> T {
        let Self { x, y } = self;
        x * x + y * y
    }

    /// The manhattan distance between two points.
    pub fn manhattan_dist(self, city: Point2d<T>) -> T {
        let diff = city - self;
        diff.manhattan_len()
    }

    /// The manhattan distance to the origin.
    pub fn manhattan_len(&self) -> T {
        self.x.abs() + self.y.abs()
    }
}

impl From<Point2d<NonZeroU16>> for Point2d {
    fn from(value: Point2d<NonZeroU16>) -> Self {
        Self {
            x: value.x.get().into(),
            y: value.y.get().into(),
        }
    }
}

impl Point2d<i64> {
    /// Subtract two points element wise.
    pub const fn sub(mut self, rhs: Point2d) -> Point2d {
        self.x -= rhs.x;
        self.y -= rhs.y;
        self
    }

    /// Multiply two points element wise.
    pub const fn mul(mut self, rhs: Point2d) -> Point2d {
        self.x *= rhs.x;
        self.y *= rhs.y;
        self
    }

    /// Get the bytes of this point in native byte order.
    pub fn to_ne_bytes(&self) -> [u8; 16] {
        let mut array = [0; 16];
        for (dest, src) in array
            .iter_mut()
            .zip(self.x.to_ne_bytes().into_iter().chain(self.y.to_ne_bytes()))
        {
            *dest = src;
        }
        array
    }
}

impl<T: DivAssign + Copy> Div<T> for Point2d<T> {
    type Output = Self;
    fn div(mut self, rhs: T) -> Self::Output {
        self /= rhs;
        self
    }
}

impl<T: DivAssign + Copy> DivAssign<T> for Point2d<T> {
    fn div_assign(&mut self, rhs: T) {
        self.x /= rhs;
        self.y /= rhs;
    }
}

impl<T: MulAssign + Copy> Mul<T> for Point2d<T> {
    type Output = Self;
    fn mul(mut self, rhs: T) -> Self::Output {
        self *= rhs;
        self
    }
}

impl<T: MulAssign + Copy> MulAssign<T> for Point2d<T> {
    fn mul_assign(&mut self, rhs: T) {
        self.x *= rhs;
        self.y *= rhs;
    }
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
/// A rectangle that includes the minimum and maximum values
pub struct Bounds<T = i64> {
    /// The corner closest to the origin.
    pub min: Point2d<T>,
    /// The corner furthest away from the origin.
    pub max: Point2d<T>,
}

impl<T: std::fmt::Debug> std::fmt::Debug for Bounds<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}..={:?}", self.min, self.max)
    }
}

impl<T: Copy + PartialEq + PartialOrd + SampleUniform> Bounds<T> {
    /// Generate a point within the bounds.
    pub fn sample<R: RngCore + ?Sized>(self, rng: &mut R) -> Point2d<T> {
        Point2d {
            x: (self.min.x..self.max.x).sample_single(rng),
            y: (self.min.y..self.max.y).sample_single(rng),
        }
    }

    /// Apply a closure to both `min` and `max`
    pub fn map<U>(&self, f: impl Fn(Point2d<T>) -> Point2d<U>) -> Bounds<U> {
        Bounds {
            min: f(self.min),
            max: f(self.max),
        }
    }
}

impl<T: Copy> Bounds<T> {
    /// Bounds at a single point with zero width and height.
    pub fn point(point: Point2d<T>) -> Self {
        Self {
            min: point,
            max: point,
        }
    }
}

impl<T: PartialOrd + Num + Copy + AddAssign> Bounds<T> {
    /// Iterate over all integer points within these bounds.
    pub fn iter(self) -> impl Iterator<Item = Point2d<T>> {
        let mut current = self.min;
        std::iter::from_fn(move || {
            if current.y > self.max.y {
                None
            } else {
                let item = current;
                current.x += T::ONE;
                if current.x > self.max.x {
                    current.x = self.min.x;
                    current.y += T::ONE;
                }
                Some(item)
            }
        })
    }
}

impl<T: Copy + Num + Add<Output = T> + Sub<Output = T> + DivAssign<T>> Bounds<T> {
    /// The middle point of these bounds.
    pub fn center(&self) -> Point2d<T> {
        (self.max - self.min) / T::TWO + self.min
    }
}

impl<T: Copy + Add<Output = T> + Sub<Output = T>> Bounds<T> {
    /// Add padding on all sides.
    pub fn pad(&self, padding: Point2d<T>) -> Self {
        Self {
            min: self.min - padding,
            max: self.max + padding,
        }
    }
}

#[cfg(test)]
#[test]
fn iter() {
    let grid = Bounds {
        min: Point2d::new(10, 42),
        max: Point2d::new(12, 43),
    };
    let mut iter = grid.iter();
    assert_eq!(iter.next(), Some(grid.min));
    assert_eq!(iter.next(), Some(Point2d::new(11, 42)));
    assert_eq!(iter.next(), Some(Point2d::new(12, 42)));
    assert_eq!(iter.next(), Some(Point2d::new(10, 43)));
    assert_eq!(iter.next(), Some(Point2d::new(11, 43)));
    assert_eq!(iter.next(), Some(Point2d::new(12, 43)));
    assert_eq!(iter.next(), None);
    assert_eq!(iter.next(), None);
}

#[cfg(test)]
#[test]
fn iter_point() {
    let grid = Bounds::point(Point2d::new(10, 42));
    let mut iter = grid.iter();
    assert_eq!(iter.next(), Some(grid.min));
    assert_eq!(iter.next(), None);
    assert_eq!(iter.next(), None);
}

impl<T: DivAssign + Copy> Div<Point2d<T>> for Bounds<T> {
    type Output = Self;
    fn div(mut self, rhs: Point2d<T>) -> Self::Output {
        self.min /= rhs;
        self.max /= rhs;
        self
    }
}

/// A helper trait for specifying generic numeric types.
pub trait Num:
    Sized
    + Copy
    + AddAssign
    + SubAssign
    + Ord
    + Sub<Output = Self>
    + Neg<Output = Self>
    + Eq
    + Add<Output = Self>
{
    /// The neutral value for addition and subtraction.
    const ZERO: Self;
    /// The neutral value for multiplication and division.
    const ONE: Self;
    /// For when you can't use `+` in const contexts, but need a `2`
    const TWO: Self;
    /// Iterate over a range. Workaround to [std::ops::Range]'s [Iterator] impl
    /// not being implementable for custom types.
    fn iter_range(range: std::ops::Range<Self>) -> impl Iterator<Item = Self>;
    /// Convert the value to a [u64]. Used for seeding random number generators
    /// from coordinates.
    fn as_u64(self) -> u64;
}

impl Num for i64 {
    const ZERO: i64 = 0;
    const ONE: i64 = 1;
    const TWO: i64 = 2;

    fn iter_range(range: std::ops::Range<Self>) -> impl Iterator<Item = Self> {
        range
    }

    fn as_u64(self) -> u64 {
        self as u64
    }
}
