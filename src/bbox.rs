use crate::shapes::{ArchivedShape, Point};
use rkyv::{Archive, Deserialize, Serialize};

pub trait CalculateBoundingBox {
    fn bbox(&self) -> BoundingBox;
}

#[derive(Debug)]
pub struct UnvalidatedBoundingBox {
    pub min: Point,
    pub max: Point,
}

impl UnvalidatedBoundingBox {
    pub fn invalid() -> Self {
        UnvalidatedBoundingBox {
            min: Point {
                x: i32::MAX,
                y: i32::MAX,
            },
            max: Point {
                x: i32::MIN,
                y: i32::MIN,
            },
        }
    }
    pub fn update(&mut self, shape: &ArchivedShape) {
        match shape {
            ArchivedShape::Rect(r) => {
                self.min.x = r.p0.x.min(self.min.x);
                self.min.y = r.p0.y.min(self.min.y);
                self.max.x = r.p0.x.max(self.max.x);
                self.max.y = r.p0.y.max(self.max.y);
                self.min.x = r.p1.x.min(self.min.x);
                self.min.y = r.p1.y.min(self.min.y);
                self.max.x = r.p1.x.max(self.max.x);
                self.max.y = r.p1.y.max(self.max.y);
            }
            ArchivedShape::Poly(p) => {
                for pt in p.points.iter() {
                    self.min.x = pt.x.min(self.min.x);
                    self.min.y = pt.y.min(self.min.y);
                    self.max.x = pt.x.max(self.max.x);
                    self.max.y = pt.y.max(self.max.y);
                }
            }
            ArchivedShape::Path(p) => {
                let pts = p.as_poly();

                for pt in pts.iter() {
                    self.min.x = pt.x.min(self.min.x);
                    self.min.y = pt.y.min(self.min.y);
                    self.max.x = pt.x.max(self.max.x);
                    self.max.y = pt.y.max(self.max.y);
                }
            }
        }
    }
    pub fn update_bbox(&mut self, bbox: &UnvalidatedBoundingBox) {
        self.min.x = self.min.x.min(bbox.min.x);
        self.min.y = self.min.y.min(bbox.min.y);
        self.max.x = self.max.x.max(bbox.max.x);
        self.max.y = self.max.y.max(bbox.max.y);
    }
}

#[derive(
    Debug,
    Eq,
    PartialEq,
    Archive,
    Deserialize,
    Serialize,
    Clone,
    Copy,
    serde::Serialize,
    serde::Deserialize,
)]
#[archive(compare(PartialEq))]
#[archive_attr(derive(Debug))]
pub struct BoundingBox {
    min: Point,
    max: Point,
}

impl BoundingBox {
    pub fn new(unvalidated: UnvalidatedBoundingBox) -> Self {
        let mut bbox = BoundingBox {
            min: Point::default(),
            max: Point::default(),
        };

        assert_ne!(
            unvalidated.min.x, unvalidated.max.x,
            "Boundingbox min x and max x mut not be equal"
        );
        if unvalidated.max.x < unvalidated.min.x {
            bbox.min.x = unvalidated.max.x;
            bbox.max.x = unvalidated.min.x;
        } else {
            bbox.min.x = unvalidated.min.x;
            bbox.max.x = unvalidated.max.x;
        }

        assert_ne!(
            unvalidated.min.y, unvalidated.max.y,
            "Boundingbox min y and max y mut not be equal"
        );
        if unvalidated.max.y < unvalidated.min.y {
            bbox.min.y = unvalidated.max.y;
            bbox.max.y = unvalidated.min.y;
        } else {
            bbox.min.y = unvalidated.min.y;
            bbox.max.y = unvalidated.max.y;
        }

        bbox
    }
    pub fn union(&mut self, bbox: &Self) {
        // Take the minimum and maximum of the two bounding boxes
        self.min.x = self.min.x.min(bbox.min.x);
        self.min.y = self.min.y.min(bbox.min.y);
        self.max.x = self.max.x.max(bbox.max.x);
        self.max.y = self.max.y.max(bbox.max.y);
    }
    #[must_use = "This used to mutate the BoundingBox it was called on, it now instead returns a new BoundingBox"]
    pub fn shift(&self, p: &Point) -> Self {
        Self {
            min: self.min.shift(p),
            max: self.max.shift(p),
        }
    }
    pub fn min(&self) -> Point {
        self.min
    }
    pub fn max(&self) -> Point {
        self.max
    }
}
