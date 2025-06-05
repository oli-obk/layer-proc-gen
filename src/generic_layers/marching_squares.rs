use std::marker::PhantomData;

use crate::{
    Chunk, ChunkExt as _, GridIndex, Layer,
    debug::{Debug, DebugContent},
    rolling_grid::GridPoint,
    vec2::{Bounds, Point2d},
};

/// Chunk type that can be used to compute countours
pub trait Density: Chunk + Debug {
    /// Whether the chunk is above the threshold for being solid/within the countour.
    fn above_threshold(&self) -> bool;
}

#[derive(PartialEq, Debug)]
/// Computes contour lines that seamlessly connect to neighboring lines
pub struct LookupIndex<DensityMap> {
    /// The lookup table index
    pub idx: u8,
    phantom: PhantomData<DensityMap>,
}

impl<DensityMap> Clone for LookupIndex<DensityMap> {
    fn clone(&self) -> Self {
        Self {
            idx: self.idx.clone(),
            phantom: self.phantom.clone(),
        }
    }
}

impl<DensityMap> Default for LookupIndex<DensityMap> {
    fn default() -> Self {
        Self {
            idx: Default::default(),
            phantom: Default::default(),
        }
    }
}

impl<DensityMap: Density> Chunk for LookupIndex<DensityMap> {
    type LayerStore<T> = T;
    type Dependencies = Layer<DensityMap>;
    const SIZE: Point2d<u8> = DensityMap::SIZE;

    fn compute(d: &Self::Dependencies, index: GridPoint<Self>) -> Self {
        let get = |x, y, i| {
            (d.get(GridPoint {
                x: GridIndex::from_raw(x + index.x.0),
                y: GridIndex::from_raw(y + index.y.0),
            })
            .above_threshold() as u8)
                << i
        };
        let idx = get(0, 1, 0) | get(1, 1, 1) | get(1, 0, 2) | get(0, 0, 3);
        LookupIndex {
            idx,
            phantom: PhantomData,
        }
    }

    fn clear(raw_points: &Self::Dependencies, index: GridPoint<Self>) {
        raw_points.clear(Self::bounds(index));
    }
}

impl<DensityMap> Debug for LookupIndex<DensityMap> {
    fn debug(&self, bounds: Bounds) -> Vec<DebugContent> {
        vec![DebugContent::Text {
            pos: bounds.center(),
            label: self.idx.to_string(),
        }]
    }
}

#[derive(PartialEq, Debug)]
/// Contour lines that seamlessly connect to neighboring lines
pub struct ContourLines<DensityMap> {
    /// zero to two shapes
    pub shapes: [Vec<Point2d>; 2],
    phantom: PhantomData<DensityMap>,
}

impl<DensityMap> Clone for ContourLines<DensityMap> {
    fn clone(&self) -> Self {
        Self {
            shapes: self.shapes.clone(),
            phantom: self.phantom.clone(),
        }
    }
}

impl<DensityMap> Default for ContourLines<DensityMap> {
    fn default() -> Self {
        Self {
            shapes: Default::default(),
            phantom: Default::default(),
        }
    }
}

impl<DensityMap: Density> Chunk for ContourLines<DensityMap> {
    type LayerStore<T> = T;
    type Dependencies = Layer<LookupIndex<DensityMap>>;
    const SIZE: Point2d<u8> = DensityMap::SIZE;

    fn compute(d: &Self::Dependencies, index: GridPoint<Self>) -> Self {
        let d = d.get(index.into_same_chunk_size()).idx;
        let bounds = Self::bounds(index);
        let l = bounds.min.x;
        let r = bounds.max.x;
        let t = bounds.min.y;
        let b = bounds.max.y;
        let c = bounds.center();
        let l = Point2d::new(l, c.y);
        let r = Point2d::new(r, c.y);
        let t = Point2d::new(c.x, t);
        let b = Point2d::new(c.x, b);
        let shapes = [
            [vec![], vec![]],
            [vec![l, b, bounds.left_bottom()], vec![]],
            [vec![b, r, bounds.right_bottom()], vec![]],
            [
                vec![l, r, bounds.right_bottom(), bounds.left_bottom()],
                vec![],
            ],
            [vec![t, bounds.right_top(), r], vec![]],
            [
                vec![l, b, bounds.left_bottom()],
                vec![t, bounds.right_top(), r],
            ],
            [
                vec![t, bounds.right_top(), bounds.right_bottom(), b],
                vec![],
            ],
            [
                vec![
                    t,
                    bounds.right_top(),
                    bounds.right_bottom(),
                    bounds.left_bottom(),
                    l,
                ],
                vec![],
            ],
            [vec![bounds.left_top(), t, l], vec![]],
            [vec![bounds.left_top(), t, b, bounds.left_bottom()], vec![]],
            [
                vec![bounds.left_top(), t, l],
                vec![bounds.right_bottom(), b, r],
            ],
            [
                vec![
                    bounds.left_top(),
                    t,
                    r,
                    bounds.right_bottom(),
                    bounds.left_bottom(),
                ],
                vec![],
            ],
            [vec![bounds.left_top(), bounds.right_top(), r, l], vec![]],
            [
                vec![
                    bounds.left_top(),
                    bounds.right_top(),
                    r,
                    b,
                    bounds.left_bottom(),
                ],
                vec![],
            ],
            [
                vec![
                    bounds.left_top(),
                    bounds.right_top(),
                    bounds.right_bottom(),
                    b,
                    l,
                ],
                vec![],
            ],
            [
                vec![
                    bounds.left_top(),
                    bounds.right_top(),
                    bounds.right_bottom(),
                    bounds.left_bottom(),
                ],
                vec![],
            ],
        ][d as usize]
            .clone();
        ContourLines {
            shapes,
            phantom: PhantomData,
        }
    }

    fn clear(raw_points: &Self::Dependencies, index: GridPoint<Self>) {
        raw_points.clear(Self::bounds(index));
    }
}

impl<DensityMap> Debug for ContourLines<DensityMap> {
    fn debug(&self, _bounds: Bounds) -> Vec<DebugContent> {
        self.shapes
            .iter()
            .filter(|shape| !shape.is_empty())
            .flat_map(|shape| {
                shape
                    .windows(2)
                    .map(|points| [points[0], points[1]])
                    .chain(Some([shape[0], *shape.last().unwrap()]))
                    .map(|[a, b]| DebugContent::Line(a.to(b)))
            })
            .collect()
    }
}
