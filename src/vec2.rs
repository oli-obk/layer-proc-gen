use std::num::NonZeroU16;

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Point2d<T: Copy = i64> {
    pub x: T,
    pub y: T,
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
