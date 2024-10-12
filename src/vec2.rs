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
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub},
};

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
)]
#[mul(forward)]
#[div(forward)]
#[mul_assign(forward)]
#[div_assign(forward)]
pub struct Point2d<T = i64> {
    pub x: T,
    pub y: T,
}

#[derive(PartialEq, Debug, Copy, Clone)]
pub struct Line<T = i64> {
    pub start: Point2d<T>,
    pub end: Point2d<T>,
}

impl Line {
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

    pub fn bounds(&self) -> GridBounds {
        GridBounds {
            min: self.start,
            max: self.end,
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

    pub fn to(self, other: Self) -> Line<T> {
        Line {
            start: self,
            end: other,
        }
    }
}

impl<T: Copy + Sub<Output = T> + Mul<Output = T> + Add<Output = T>> Point2d<T> {
    pub fn dist_squared(self, center: Point2d<T>) -> T {
        (self - center).len_squared()
    }
    pub fn len_squared(self) -> T {
        let Self { x, y } = self;
        x * x + y * y
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
    pub const fn rem_euclid(&self, divisor: Point2d<u8>) -> Point2d<usize> {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "remainder op with a u8 will alway fit in usize"
        )]
        let x = self.x.rem_euclid(divisor.x as i64) as usize;
        #[expect(
            clippy::cast_possible_truncation,
            reason = "remainder op with a u8 will alway fit in usize"
        )]
        let y = self.y.rem_euclid(divisor.y as i64) as usize;
        Point2d { x, y }
    }

    pub const fn div_euclid(&self, divisor: Point2d<NonZeroU16>) -> Self {
        Point2d {
            x: self.x.div_euclid(divisor.x.get() as i64),
            y: self.y.div_euclid(divisor.y.get() as i64),
        }
    }

    pub const fn sub(mut self, rhs: Point2d) -> Point2d {
        self.x -= rhs.x;
        self.y -= rhs.y;
        self
    }

    pub const fn mul(mut self, rhs: Point2d<NonZeroU16>) -> Point2d {
        self.x *= rhs.x.get() as i64;
        self.y *= rhs.y.get() as i64;
        self
    }

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

#[test]
fn rem_euclid() {
    assert_eq!(
        Point2d { x: -3_i64, y: -3 }.rem_euclid(Point2d { x: 16, y: 16 }),
        Point2d { x: 13, y: 13 }
    );
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
pub struct GridBounds<T = i64> {
    pub min: Point2d<T>,
    pub max: Point2d<T>,
}

impl<T: std::fmt::Debug> std::fmt::Debug for GridBounds<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}..={:?}", self.min, self.max)
    }
}

impl<T: Copy + PartialEq + PartialOrd + SampleUniform> GridBounds<T> {
    pub fn sample<R: RngCore + ?Sized>(self, rng: &mut R) -> Point2d<T> {
        Point2d {
            x: (self.min.x..self.max.x).sample_single(rng),
            y: (self.min.y..self.max.y).sample_single(rng),
        }
    }

    /// Apply a closure to both `min` and `max`
    pub fn map<U>(&self, f: impl Fn(Point2d<T>) -> Point2d<U>) -> GridBounds<U> {
        GridBounds {
            min: f(self.min),
            max: f(self.max),
        }
    }
}

impl<T: Copy> GridBounds<T> {
    pub fn point(point: Point2d<T>) -> Self {
        Self {
            min: point,
            max: point,
        }
    }
}

impl<T: PartialOrd + Num + Copy + AddAssign> GridBounds<T> {
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

impl<T: Copy + Num + Add<Output = T> + Sub<Output = T> + DivAssign<T>> GridBounds<T> {
    pub fn center(&self) -> Point2d<T> {
        (self.max - self.min) / T::TWO + self.min
    }
}

impl GridBounds {
    /// Add padding on all sides.
    pub fn pad(&self, padding: Point2d) -> Self {
        Self {
            min: self.min - padding,
            max: self.max + padding,
        }
    }
}

#[cfg(test)]
#[test]
fn iter() {
    let grid = GridBounds {
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
    let grid = GridBounds::point(Point2d::new(10, 42));
    let mut iter = grid.iter();
    assert_eq!(iter.next(), Some(grid.min));
    assert_eq!(iter.next(), None);
    assert_eq!(iter.next(), None);
}

impl<T: DivAssign + Copy> Div<Point2d<T>> for GridBounds<T> {
    type Output = Self;
    fn div(mut self, rhs: Point2d<T>) -> Self::Output {
        self.min /= rhs;
        self.max /= rhs;
        self
    }
}

pub trait Num {
    const ONE: Self;
    const TWO: Self;
}

impl Num for i64 {
    const ONE: i64 = 1;
    const TWO: i64 = 1;
}
