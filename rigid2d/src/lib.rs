use glam::Vec2;

#[derive(Default, Debug)]
pub struct Body {
    pub position: Vec2,
    pub velocity: Vec2,

    pub rotation: f32,
    pub angular_velocity: f32,
}

impl Body {
    /// Impulse directional vector magnitude would increase the
    /// acceleration by that much if applied to the center of the object (pos `vec2(0.0, 0.0)`).
    /// If applied offset, some of the impulse goes to rotation.
    pub fn add_impulse(&mut self, pos: Vec2, dir: Vec2) {
        self.velocity += dir;
        self.angular_velocity += pos.perp_dot(dir);
    }

    pub fn step(&mut self, dt: f32) {
        self.position += self.velocity * dt;
        self.rotation += self.angular_velocity * dt;
    }
}
