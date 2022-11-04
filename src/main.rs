use rkyv::{
    archived_root,
    ser::{
        serializers::{AllocScratch, CompositeSerializer, ScratchTracker, WriteSerializer},
        Serializer,
    },
    Deserialize, Infallible,
};

mod bbox;
mod shapes;

use bbox::{BoundingBox, CalculateBoundingBox, UnvalidatedBoundingBox};

use shapes::{
    bbbox, ArchivedPath, ArchivedPoint, ArchivedPoly, ArchivedRect, ArchivedShape, ArchivedShapes,
    Point,
};

use bincode;

use memmap2::Mmap;
use shapes::{Path, Poly, Rect, Shape, Shapes};

use std::io::{BufWriter, Read, Write};
use std::{
    collections::HashMap,
    fs::{File, OpenOptions},
};

use geo::Intersects;

pub type GeoRect = geo::Rect<i64>;
pub type GeoPolygon = geo::Polygon<i64>;

#[derive(Debug, Clone)]
pub enum GeoShapeEnum {
    Rect(GeoRect),
    Polygon(GeoPolygon),
}

#[derive(Debug)]
pub struct Tile {
    pub extents: GeoRect,
    pub shapes: Vec<usize>,
}

fn main() {
    let t = std::time::Instant::now();

    // process_vlsir();

    // read_rkyv_calc_bbox();

    len_polys_paths();

    // #[cfg(feature = "rkyv_64")]
    // create_tile_map();
    // #[cfg(feature = "rkyv_32")]
    // read_bincode_write_rkyv();

    println!("{:?}", t.elapsed());
}

fn read_rkyv_calc_bbox() -> usize {
    let f = File::open("test64x64.rkyv").unwrap();

    let mmap = unsafe { Mmap::map(&f).unwrap() };

    let archived_shapes = unsafe { archived_root::<Shapes>(&mmap) };

    println!("Stored BBox: {:?}", archived_shapes.bbox);

    let bbox = bbbox(std::sync::Arc::new(&archived_shapes.shapes));

    println!("{bbox:?}");

    archived_shapes.shapes.len()
}

#[cfg(feature = "rkyv_64")]
fn create_tile_map() {
    let f = File::open("test.rkyv").unwrap();

    let mmap = unsafe { Mmap::map(&f).unwrap() };

    let archived_shapes = unsafe { archived_root::<Shapes>(&mmap) };

    let bbox: BoundingBox = archived_shapes
        .bbox
        .deserialize(&mut rkyv::Infallible)
        .unwrap();

    println!("TileMap extents: {bbox:?}");

    let tilemap_shift = Point {
        x: -bbox.min().x,
        y: -bbox.min().y,
    };

    println!("tilemap_shift {tilemap_shift:?}");

    let dx = u32::try_from(bbox.max().x - bbox.min().x).unwrap();
    let dy = u32::try_from(bbox.max().y - bbox.min().y).unwrap();

    let extents: u32 = 16384;

    let num_x_tiles = (dx as f32 / extents as f32).ceil() as u32;
    let num_y_tiles = (dy as f32 / extents as f32).ceil() as u32;

    // println!("dx {dx} dy {dy} num_x_tiles {num_x_tiles} num_y_tiles {num_y_tiles}");

    let mut x = bbox.min().x;
    let mut y = bbox.min().y;

    let mut tilemap = HashMap::<(u32, u32), Tile>::new();

    for iy in 0..num_y_tiles {
        let ymin = y;
        y += extents as i32;
        let ymax = y;
        for ix in 0..num_x_tiles {
            let xmin = x;
            x += extents as i32;
            let xmax = x;

            let extents = GeoRect::new((xmin as i64, ymin as i64), (xmax as i64, ymax as i64));

            tilemap.insert(
                (ix, iy),
                Tile {
                    extents,
                    shapes: vec![],
                },
            );
        }

        x = bbox.min().x;
    }

    let mut shape_count = 0;

    for i in 0..archived_shapes.shapes.len() {
        let s = &archived_shapes.shapes[i];

        // println!("{s:?}");

        // let mut bbox = s.bbox();

        // println!("pre-shift: {bbox:?}");

        let bbox = s.bbox().shift(&tilemap_shift);

        // println!("post-shift: {bbox:?}");

        // let min_tile_x = (bbox.min.x / extents as i32).min(0) as u32;
        // let max_tile_x = (bbox.max.x / extents as i32).max(num_x_tiles) as u32;
        // let min_tile_y = (bbox.min.y / extents as i32).min(0) as u32;
        // let max_tile_y = (bbox.max.y / extents as i32).max(num_y_tiles) as u32;

        let min_tile_x = bbox.min().x as u32 / extents;
        let min_tile_y = bbox.min().y as u32 / extents;
        let max_tile_x = bbox.max().x as u32 / extents;
        let max_tile_y = bbox.max().y as u32 / extents;

        let geo_shape = match s {
            ArchivedShape::Rect(r) => {
                // println!("{r:?}");
                let ArchivedRect { p0, p1, .. } = r;

                let xmin = p0.x as i64;
                let ymin = p0.y as i64;
                let xmax = p1.x as i64;
                let ymax = p1.y as i64;

                let rect = GeoRect::new((xmin, ymin), (xmax, ymax));

                let geo_shape = GeoShapeEnum::Rect(rect);

                geo_shape
            }
            ArchivedShape::Poly(p) => {
                // println!("{p:?}");
                let poly = GeoPolygon::new(
                    p.points.iter().map(|p| (p.x as i64, p.y as i64)).collect(),
                    vec![],
                );

                GeoShapeEnum::Polygon(poly)
            }
            ArchivedShape::Path(p) => {
                // println!("{p:?}");

                let p = p.as_poly();

                let poly = GeoPolygon::new(
                    p.into_iter().map(|p| (p.x as i64, p.y as i64)).collect(),
                    vec![],
                );

                GeoShapeEnum::Polygon(poly)
            }
        };

        // println!("min_tile_x: {min_tile_x}, max_tile_x: {max_tile_x}, min_tile_y: {min_tile_y}, max_tile_y: {max_tile_y}");
        // println!("{geo_shape:?}");
        // println!("{:?}", tilemap.get(&(min_tile_x, min_tile_y)));

        // panic!();

        // std::thread::sleep(std::time::Duration::from_secs(5));

        for x in min_tile_x..(max_tile_x + 1) {
            for y in min_tile_y..(max_tile_y + 1) {
                // println!("tile x {x}, y {y}");
                let Tile { extents, shapes } = tilemap.get_mut(&(x, y)).unwrap();

                let extents = &*extents;

                // println!("extents {extents:?}");

                match &geo_shape {
                    GeoShapeEnum::Rect(r) => {
                        // println!("extents {extents:?}");
                        // println!("rect:   {r:?}");
                        if r.intersects(extents) {
                            shapes.push(i);
                            // println!("pushing idx [{i}] to tile {extents:?}");
                        }
                    }
                    GeoShapeEnum::Polygon(p) => {
                        if p.intersects(extents) {
                            shapes.push(i);
                            // println!("pushing idx [{i}] to tile {extents:?}");
                        }
                    }
                }
            }
        }

        // println!("shape_count: {shape_count}");

        shape_count += 1;

        if shape_count % 1_000_000 == 0 {
            println!("shapes processed: {shape_count}");
        }
    }

    let mut shapes_to_archive = vec![];

    let min_x_tile = 56;
    let max_x_tile = 119;
    // let max_x_tile = 57;
    let min_y_tile = 204;
    let max_y_tile = 267;
    // let max_y_tile = 207;

    for (ix, x) in (min_x_tile..=max_x_tile).enumerate() {
        for (iy, y) in (min_y_tile..=max_y_tile).enumerate() {
            let Tile { shapes, extents } = tilemap.get(&(x, y)).unwrap();
            // println!("tile x {x}({ix}), y {y}({iy}), extents: {extents:?}");

            for &idx in shapes {
                let shape: Shape = archived_shapes.shapes[idx]
                    .deserialize(&mut Infallible)
                    .unwrap();
                shapes_to_archive.push(shape);
            }
        }
    }

    // let bbox = shapes_to_archive.bbox();

    let min = tilemap.get(&(min_x_tile, min_y_tile)).unwrap().extents;
    let min = Point {
        x: min.min().x as i32,
        y: min.min().y as i32,
    };

    let max = tilemap.get(&(max_x_tile, max_y_tile)).unwrap().extents;
    let max = Point {
        x: max.max().x as i32,
        y: max.max().y as i32,
    };

    let unvalidated = UnvalidatedBoundingBox { min, max };
    println!("{unvalidated:?}");
    let bbox = BoundingBox::new(unvalidated);
    println!("{bbox:?}");

    let dx = bbox.max().x - bbox.min().x;
    let dy = bbox.max().y - bbox.min().y;

    let tiles_x = dx / extents as i32;
    let tiles_y = dy / extents as i32;

    println!("dx: {dx}, dy: {dy}, tiles_x: {tiles_x}, tiles_y: {tiles_y}");

    println!("64x64 tile bbox {bbox:?}");

    let shapes_to_archive = Shapes {
        bbox,
        shapes: shapes_to_archive,
    };

    let f = OpenOptions::new()
        .write(true)
        .create(true)
        .open("test64x64.bincode")
        .unwrap();

    let encoded: Vec<u8> = bincode::serialize(&shapes_to_archive).unwrap();

    let mut bw = BufWriter::new(&f);

    bw.write(encoded.as_slice()).unwrap();
}

#[cfg(feature = "rkyv_32")]
fn read_bincode_write_rkyv() {
    let mut f = File::open("test64x64.bincode").unwrap();

    let mut encoded = vec![];

    f.read_to_end(&mut encoded).unwrap();

    let shapes_to_archive: Shapes = bincode::deserialize(&encoded[..]).unwrap();

    // println!("{decoded:?}");

    let f = OpenOptions::new()
        .write(true)
        .create(true)
        .open("test64x64.rkyv")
        .unwrap();

    let bw = BufWriter::new(&f);

    let mut serializer = CompositeSerializer::new(
        WriteSerializer::new(bw),
        ScratchTracker::new(AllocScratch::default()),
        Infallible,
    );

    serializer.serialize_value(&shapes_to_archive).unwrap();
}

fn test_read_rkyv() {
    let f = File::open("test.rkyv").unwrap();

    let mmap = unsafe { Mmap::map(&f).unwrap() };

    let archived_value = unsafe { archived_root::<Shapes>(&mmap) };

    let t = std::time::Instant::now();
    for (i, _) in archived_value.shapes.iter().enumerate() {
        if i % 10_000_000 == 0 {
            println!(
                "done: {i}, rate: {}/s",
                i as f64 / t.elapsed().as_secs_f64()
            )
        }
        // print_shape(s);
    }
    println!("Total duration: {:?}", t.elapsed());
}

fn len_polys_paths() {
    let f = File::open("test64x64.rkyv").unwrap();

    let mmap = unsafe { Mmap::map(&f).unwrap() };

    let archived_value = unsafe { archived_root::<Shapes>(&mmap) };

    let mut polys_lens = Vec::with_capacity(13446559);

    let mut paths_lens = Vec::with_capacity(3774369);

    for s in archived_value.shapes.iter() {
        match s {
            ArchivedShape::Poly(p) => {
                polys_lens.push(p.points.len());
            }
            ArchivedShape::Path(p) => {
                paths_lens.push(p.points.len());
            }
            _ => (),
        }
    }

    let polys_min = *polys_lens.iter().min().unwrap();
    let polys_max = *polys_lens.iter().max().unwrap();
    let polys_avg = polys_lens.iter().sum::<usize>() as f32 / polys_lens.len() as f32;
    let len = polys_lens.len();
    polys_lens.sort_unstable();
    let polys_median = polys_lens[len / 2];

    let paths_min = *paths_lens.iter().min().unwrap();
    let paths_max = *paths_lens.iter().max().unwrap();
    let paths_avg = paths_lens.iter().sum::<usize>() as f32 / paths_lens.len() as f32;
    let len = paths_lens.len();
    let paths_lens_5 = paths_lens
        .iter()
        .filter(|&x| *x == 2)
        .collect::<Vec<_>>()
        .len();

    println!("Num paths 5: {paths_lens_5}");

    paths_lens.sort_unstable();
    let paths_median = paths_lens[len / 2];

    println!(
        "polys_lens: min {polys_min}, max {polys_max}, avg {polys_avg}, median {polys_median}"
    );
    println!(
        "paths_lens: min {paths_min}, max {paths_max}, avg {paths_avg}, median {paths_median}"
    );
}

fn shapes_count() {
    let f = File::open("test.rkyv").unwrap();

    let mmap = unsafe { Mmap::map(&f).unwrap() };

    let archived_value = unsafe { archived_root::<Shapes>(&mmap) };

    let mut rect_count = 0;

    let mut polys_count = 0;

    let mut paths_count = 0;

    for s in archived_value.shapes.iter() {
        match s {
            ArchivedShape::Poly(_) => {
                // polys_lens.push(p.points.len());
                polys_count += 1;
            }
            ArchivedShape::Path(_) => {
                // paths_lens.push(p.points.len());
                paths_count += 1;
            }
            ArchivedShape::Rect(_) => rect_count += 1,
        }
    }

    println!("rects {rect_count}, polys {polys_count}, paths {paths_count}");
}

fn create_rkvy() {
    let shapes = process_vlsir();

    let f = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open("test.rkyv")
        .unwrap();

    let bw = BufWriter::new(&f);

    let mut serializer = CompositeSerializer::new(
        WriteSerializer::new(bw),
        ScratchTracker::new(AllocScratch::default()),
        Infallible,
    );

    serializer.serialize_value(&shapes).unwrap();

    let tracker = serializer.into_components().1;

    println!("max_bytes_allocated: {}, max_allocations: {}, max_alignment: {}, min_buffer_size: {}, min_buffer_size_max_error: {}",
        tracker.max_bytes_allocated(),
        tracker.max_allocations(),
        tracker.max_alignment(),
        tracker.min_buffer_size(),
        tracker.min_buffer_size_max_error(),
    );

    let mmap = unsafe { Mmap::map(&f).unwrap() };

    let archived_shapes = unsafe { archived_root::<Shapes>(&mmap) };

    let bbox: BoundingBox = archived_shapes.bbox.deserialize(&mut Infallible).unwrap();

    println!("Design extents: {:?}", bbox);

    println!("{:?}", &archived_shapes.shapes[0]);
}

use layout21::raw::{
    self, proto::ProtoImporter, BoundBoxTrait, Element, Layers, Layout, LayoutResult, Transform,
    TransformTrait,
};

fn process_vlsir() -> Shapes {
    let plib = raw::proto::proto::open(
        "/home/colepoirier/Dropbox/rust_2020_onwards/doug/doug/libs/oscibear.proto",
    )
    .unwrap();
    let lib = ProtoImporter::import(&plib, None).unwrap();

    let cell_ptr = lib.cells.iter().last().unwrap();

    let cell = cell_ptr.read().unwrap();

    let layout = cell.layout.as_ref().unwrap();

    let lib_layers = &lib.layers;

    flatten(layout, lib_layers).unwrap()
}

/// Flatten a [Layout], particularly its hierarchical instances, to a vector of [Element]s
pub fn flatten(layout: &Layout, lib_layers: &layout21::utils::Ptr<Layers>) -> LayoutResult<Shapes> {
    // Kick off recursive calls, with the identity-transform applied for the top-level `layout`

    let vlsir = layout.flatten().unwrap();

    let mut set = std::collections::HashSet::<&Element>::with_capacity(vlsir.len());

    let mut num_duplicates = 0;

    for e in vlsir.iter() {
        if !set.insert(e) {
            // println!("Found duplicate element! {e:?}");
            num_duplicates += 1;
        }
    }

    println!("Num duplicates: {num_duplicates}");

    panic!();

    let mut shapes = Vec::new();
    let mut bbox = raw::BoundBox::default();
    flatten_helper(
        layout,
        &Transform::identity(),
        &mut bbox,
        &mut shapes,
        lib_layers,
    )
    .unwrap();

    let mut set = std::collections::HashSet::<&Shape>::with_capacity(shapes.len());

    for s in shapes.iter() {
        if !set.insert(s) {
            println!("Found duplicate shape! {s:?}");
        }
    }

    assert!(!bbox.is_empty(), "bbox must be valid!");

    let unvalidated = UnvalidatedBoundingBox {
        min: Point {
            x: bbox.p0.x as i32,
            y: bbox.p0.y as i32,
        },
        max: {
            Point {
                x: bbox.p1.x as i32,
                y: bbox.p1.y as i32,
            }
        },
    };

    let bbox = BoundingBox::new(unvalidated);

    Ok(Shapes { bbox, shapes })
}

fn flatten_helper(
    layout: &Layout,
    trans: &Transform,
    bbox: &mut raw::BoundBox,
    shapes: &mut Vec<Shape>,
    lib_layers: &layout21::utils::Ptr<Layers>,
) -> LayoutResult<()> {
    let layers = &lib_layers.read().unwrap().slots;

    // Translate each geometric element
    for elem in layout.elems.iter() {
        let layer = layers.get(elem.layer).unwrap().layernum as u8;
        let mut elem = elem.clone();
        // And translate the inner shape by `trans`
        elem.inner = elem.inner.transform(trans);
        *bbox = elem.inner.union(bbox);
        let shape = match elem.inner {
            raw::Shape::Rect(r) => Shape::Rect(Rect {
                p0: Point {
                    x: r.p0.x as i32,
                    y: r.p0.y as i32,
                },
                p1: Point {
                    x: r.p1.x as i32,
                    y: r.p1.y as i32,
                },
                layer,
            }),
            raw::Shape::Polygon(p) => Shape::Poly(Poly {
                points: p
                    .points
                    .iter()
                    .map(|p| Point {
                        x: p.x as i32,
                        y: p.y as i32,
                    })
                    .collect::<Vec<Point>>(),
                layer,
            }),
            raw::Shape::Path(p) => Shape::Path(Path {
                points: p
                    .points
                    .iter()
                    .map(|p| Point {
                        x: p.x as i32,
                        y: p.y as i32,
                    })
                    .collect::<Vec<Point>>(),
                width: p.width as u32,
                layer,
            }),
        };
        shapes.push(shape);
    }
    // Note text-valued "annotations" are ignored

    // Visit all of `layout`'s instances, recursively getting their elements
    for inst in &layout.insts {
        // Get the cell's layout-definition, or fail
        let cell = inst.cell.read()?;
        let layout = cell.layout.as_ref().unwrap();

        // Create a new [Transform], cascading the parent's and instance's
        let inst_trans = Transform::from_instance(&inst.loc, inst.reflect_vert, inst.angle);
        let trans = Transform::cascade(&trans, &inst_trans);

        // And recursively add its elements
        flatten_helper(&layout, &trans, bbox, shapes, lib_layers)?;
    }
    Ok(())
}
