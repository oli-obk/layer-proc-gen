use std::{
    num::NonZeroU16,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign},
};

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Point2d<T = i64> {
    pub x: T,
    pub y: T,
}

impl<T: std::fmt::Debug> std::fmt::Debug for Point2d<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        (&self.x, &self.y).fmt(f)
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
}

impl<T: Copy + SubAssign + Mul<Output = T> + Add<Output = T>> Point2d<T> {
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
}

impl<T: MulAssign> Mul<Point2d<T>> for Point2d<T> {
    type Output = Self;
    fn mul(mut self, rhs: Point2d<T>) -> Self::Output {
        self *= rhs;
        self
    }
}
impl<T: MulAssign> MulAssign for Point2d<T> {
    fn mul_assign(&mut self, rhs: Point2d<T>) {
        self.x *= rhs.x;
        self.y *= rhs.y;
    }
}

impl<T: DivAssign> Div<Point2d<T>> for Point2d<T> {
    type Output = Self;
    fn div(mut self, rhs: Point2d<T>) -> Self::Output {
        self /= rhs;
        self
    }
}
impl<T: DivAssign> DivAssign for Point2d<T> {
    fn div_assign(&mut self, rhs: Point2d<T>) {
        self.x /= rhs.x;
        self.y /= rhs.y;
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

impl<T: SubAssign> Sub<Point2d<T>> for Point2d<T> {
    type Output = Self;
    fn sub(mut self, rhs: Point2d<T>) -> Self::Output {
        self -= rhs;
        self
    }
}
impl<T: SubAssign> SubAssign for Point2d<T> {
    fn sub_assign(&mut self, rhs: Point2d<T>) {
        self.x -= rhs.x;
        self.y -= rhs.y;
    }
}

impl<T: AddAssign> Add<Point2d<T>> for Point2d<T> {
    type Output = Self;
    fn add(mut self, rhs: Point2d<T>) -> Self::Output {
        self += rhs;
        self
    }
}
impl<T: AddAssign> AddAssign for Point2d<T> {
    fn add_assign(&mut self, rhs: Point2d<T>) {
        self.x += rhs.x;
        self.y += rhs.y;
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

impl GridBounds {
    pub fn point(point: Point2d) -> Self {
        Self {
            min: point,
            max: point,
        }
    }

    pub fn iter(self) -> impl Iterator<Item = Point2d> {
        let mut current = self.min;
        std::iter::from_fn(move || {
            if current.y > self.max.y {
                None
            } else {
                let item = current;
                current.x += 1;
                if current.x > self.max.x {
                    current.x = self.min.x;
                    current.y += 1;
                }
                Some(item)
            }
        })
    }

    pub fn center(&self) -> Point2d {
        (self.max - self.min) / 2 + self.min
    }

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
