use layer_proc_gen::*;
use rolling_grid::RollingGrid;
use vec2::Point2d;

#[derive(Default)]
struct TheLayer(RollingGrid<Self>);
struct TheChunk;

impl Layer for TheLayer {
    type Chunk = TheChunk;

    fn rolling_grid(&self) -> &RollingGrid<Self> {
        &self.0
    }

    fn rolling_grid_mut(&mut self) -> &mut RollingGrid<Self> {
        &mut self.0
    }
}

impl Chunk for TheChunk {
    type Layer = TheLayer;
}

#[test]
fn create_layer() {
    let mut layer = TheLayer::default();
    layer
        .rolling_grid_mut()
        .set(Point2d { x: 42, y: 99 }, TheChunk);
}

#[test]
#[should_panic]
fn double_assign_chunk() {
    let mut layer = TheLayer::default();
    layer
        .rolling_grid_mut()
        .set(Point2d { x: 42, y: 99 }, TheChunk);
    layer
        .rolling_grid_mut()
        .set(Point2d { x: 42, y: 99 }, TheChunk);
}
