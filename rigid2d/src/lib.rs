//! A small library for handling basic 2d rigid body physics.

#![deny(missing_docs)]

use std::f32::consts::PI;

use glam::Vec2;

#[derive(Default, Debug)]
/// The object that we're moving around.
pub struct Body {
    /// Position in world coordinates. Changes automatically on [step](Self::step)
    /// based on [Self::velocity].
    pub position: Vec2,
    /// Velocity of the object. Is applied to [Self::position] on every [step](Self::step)
    /// after having the [Self::linear_forces] applied.
    pub velocity: Vec2,
    /// Summation of all acceleration forces applied since last
    /// step. Will get reset to zero at next [step](Self::step)
    pub linear_forces: Vec2,

    /// Rotation in radians.
    pub rotation: f32,
    /// Angular velocity of the object. Is applied to [Self::rotation] on every [step](Self::step)
    /// after having the [Self::angular_forces] applied.
    pub angular_velocity: f32,
    /// Summation of all angular forces applied since last
    /// step. Will get reset to zero at next [step](Self::step)
    pub angular_forces: f32,
}

impl Body {
    /// Impulse directional vector magnitude would increase the
    /// acceleration by that much if applied to the center of the object (pos `vec2(0.0, 0.0)`).
    /// If applied offset, some of the impulse goes to rotation.
    pub fn add_impulse(&mut self, pos: Vec2, dir: Vec2) {
        self.linear_forces += dir;
        if pos != Vec2::ZERO {
            self.angular_forces += pos.perp_dot(dir) * 4. / PI;
        }
    }

    /// Actually apply forces and move/rotate the object accordingly.
    pub fn step(&mut self, dt: f32) {
        self.velocity += std::mem::take(&mut self.linear_forces) * dt;
        self.angular_velocity += std::mem::take(&mut self.angular_forces) * dt;
        self.position += self.velocity * dt;
        self.rotation += self.angular_velocity * dt;
    }

    /// The velocity that the point of the object at the given offset
    /// moves. Takes into account linear and angular velocity.
    pub fn velocity_at_local_point(&self, point: Vec2) -> Vec2 {
        self.velocity + self.angular_velocity * point.perp()
    }
}
