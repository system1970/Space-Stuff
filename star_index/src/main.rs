use std::error::Error;
use std::fs::File;
use std::path::Path;

use clap::Parser;
use rstar::{RTree, RTreeObject, AABB, PointDistance};
use serde::Deserialize;

/// Struct to represent a single star entry from CSV
#[allow(dead_code)]
#[derive(Debug, Deserialize, Clone)] 
struct Star {
    obj_id: u64,
    ra: f64,
    dec: f64,
    u: Option<f64>,
    g: Option<f64>,
    r: Option<f64>,
    i: Option<f64>,
    z: Option<f64>,
}

/// Struct to wrap Star for spatial indexing
#[derive(Clone)]
struct StarPoint {
    star: Star,
}

impl RTreeObject for StarPoint {
    type Envelope = AABB<[f64; 2]>;

    fn envelope(&self) -> Self::Envelope {
        AABB::from_point([self.star.ra, self.star.dec])
    }
}

impl PointDistance for StarPoint {
    fn distance_2(&self, point: &[f64; 2]) -> f64 {
        let dx = self.star.ra - point[0];
        let dy = self.star.dec - point[1];
        dx * dx + dy * dy
    }
}

/// CLI Arguments
#[derive(Parser, Debug)]
#[command(name = "Star Indexer", version, about = "Find nearest stars using R*-tree")]
struct Args {
    /// Path to the CSV file
    #[arg(short, long)]
    file: String,

    /// RA coordinate to search from
    #[arg(short, long)]
    ra: f64,

    /// Dec coordinate to search from
    #[arg(short, long)]
    dec: f64,

    /// Number of nearest neighbors to return
    #[arg(short, long, default_value_t = 5)]
    n: usize,
}

fn load_stars_from_csv<P: AsRef<Path>>(path: P) -> Result<Vec<Star>, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false) // tell it to treat all lines as data
        .from_reader(file);

    let mut records = rdr.deserialize::<Star>();

    // Skip the first line manually
    records.next(); records.next();

    let mut stars = Vec::new();
    for result in records {
        let star: Star = result?;
        stars.push(star);
    }

    Ok(stars)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    let stars = load_stars_from_csv(&args.file)?;
    let points: Vec<StarPoint> = stars.into_iter().map(|s| StarPoint { star: s }).collect();

    let rtree = RTree::bulk_load(points);

    let nearest = rtree.nearest_neighbor_iter(&[args.ra, args.dec]).take(args.n);

    for point in nearest {
        println!(
            "obj_id: {}, RA: {}, Dec: {}",
            point.star.obj_id, point.star.ra, point.star.dec
        );
    }

    Ok(())
}