use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use clap::Parser;
use rstar::{RTree, RTreeObject, AABB, PointDistance};
use serde::Deserialize;

/// Struct to represent a single star entry from CSV.
/// Note: The field names are case-sensitive and must match the CSV headers.
#[allow(dead_code)] // u,g,r,i,z might not be used by the RTree but are part of the data
#[derive(Debug, Deserialize, Clone, PartialEq)] // Added PartialEq for testing
struct Star {
    obj_id: u64, // Make sure CSV header is "obj_id" or matches this field name if using serde rename
    ra: f64,
    dec: f64,
    u: Option<f64>,
    g: Option<f64>,
    r: Option<f64>,
    i: Option<f64>,
    z: Option<f64>,
}

/// Struct to wrap Star for spatial indexing
#[derive(Clone, Debug)] // Added Debug for testing
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
        dx * dx + dy * dy // Euclidean distance squared
    }
}

/// CLI Arguments
#[derive(Parser, Debug)]
#[command(name = "Star Indexer", version, about = "Find nearest stars using R*-tree from a CSV file.")]
struct Args {
    /// Path to the CSV file containing star data.
    #[arg(short, long)]
    file: String,

    /// RA (Right Ascension) coordinate to search from (in degrees).
    #[arg(short, long)]
    ra: f64,

    /// Dec (Declination) coordinate to search from (in degrees).
    #[arg(short, long)]
    dec: f64,

    /// Number of nearest neighbors to return.
    #[arg(short, long, default_value_t = 5)]
    n: usize,
}

/// Loads star data from a CSV file.
///
/// The function expects a CSV file where:
/// - The very first line is skipped (assumed to be a comment or BOM).
/// - The second line is treated as the header row.
/// - Subsequent lines contain star data with columns matching the `Star` struct fields
///   (e.g., `obj_id`, `ra`, `dec`).
///
/// # Arguments
///
/// * `path` - A type that implements `AsRef<Path>`, providing the path to the CSV file.
///
/// # Returns
///
/// * `Ok(Vec<Star>)` - A vector of `Star` structs if loading and parsing are successful.
/// * `Err(Box<dyn std::error::Error>)` - An error if the file cannot be opened,
///   read, or if CSV parsing fails.
fn load_stars_from_csv<P: AsRef<Path>>(path: P) -> Result<Vec<Star>, Box<dyn Error>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    // Skip the first line (e.g., BOM and #Table1 comment)
    let mut first_line = String::new();
    reader.read_line(&mut first_line)?; // Read and discard

    // Now, use the rest of the reader for csv parsing, treating the next line as header
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true) // The line after the skipped one is the header
        .from_reader(reader);

    let mut stars = Vec::new();
    for result in rdr.deserialize() { // Use deserialize() directly
        let star: Star = result?;
        stars.push(star);
    }

    Ok(stars)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    println!("Loading stars from: {}", args.file);
    let stars = match load_stars_from_csv(&args.file) {
        Ok(s) => {
            if s.is_empty() {
                eprintln!("Warning: No stars loaded from the CSV file. Ensure the file is not empty and format is correct.");
                // Optionally, exit here if no stars means no work to do
                // return Ok(()); 
            }
            println!("Loaded {} stars.", s.len());
            s
        }
        Err(e) => {
            eprintln!("Error loading stars: {}", e);
            return Err(e);
        }
    };
    
    if stars.is_empty() {
        println!("No stars to index. Exiting.");
        return Ok(());
    }

    let points: Vec<StarPoint> = stars.into_iter().map(|s| StarPoint { star: s }).collect();

    println!("Building R*-tree index...");
    let rtree = RTree::bulk_load(points);
    println!("R*-tree index built.");

    println!(
        "Searching for {} nearest neighbors to RA: {}, Dec: {}...",
        args.n, args.ra, args.dec
    );
    let nearest = rtree.nearest_neighbor_iter(&[args.ra, args.dec]).take(args.n);

    let mut count = 0;
    for point in nearest {
        count += 1;
        println!(
            "Neighbor {}: obj_id: {}, RA: {:.5}, Dec: {:.5}, Distance_sq: {:.5}",
            count,
            point.star.obj_id,
            point.star.ra,
            point.star.dec,
            point.distance_2(&[args.ra, args.dec]) // Calculate actual distance for output
        );
    }
    if count == 0 {
        println!("No neighbors found within the dataset for the given coordinates.");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*; // Import everything from the outer scope (main.rs)
    use std::io::Write; // For writing to temp files

    // Helper function to create a temporary CSV file with given content
    fn create_temp_csv(content: &str) -> Result<tempfile::NamedTempFile, Box<dyn Error>> {
        let mut temp_file = tempfile::NamedTempFile::new()?;
        writeln!(temp_file, "{}", content)?; // Use writeln to ensure newline
        Ok(temp_file)
    }

    #[test]
    fn test_load_stars_ok() {
        let csv_content = "\u{FEFF}#Table1\n\
obj_id,ra,dec,u,g,r,i,z\n\
1,150.0,2.0,18.0,17.5,17.0,16.8,16.5\n\
2,150.1,2.1,19.0,18.5,18.0,17.8,17.5\n\
3,150.2,2.2,,,,,\n"; // Test with missing optional values

        let temp_file = create_temp_csv(csv_content).expect("Failed to create temp CSV for test_load_stars_ok");
        let result = load_stars_from_csv(temp_file.path());

        assert!(result.is_ok(), "load_stars_from_csv returned an error: {:?}", result.err());
        let stars = result.unwrap();

        assert_eq!(stars.len(), 3);
        assert_eq!(stars[0], Star {
            obj_id: 1, ra: 150.0, dec: 2.0, u: Some(18.0), g: Some(17.5), r: Some(17.0), i: Some(16.8), z: Some(16.5)
        });
        assert_eq!(stars[1], Star {
            obj_id: 2, ra: 150.1, dec: 2.1, u: Some(19.0), g: Some(18.5), r: Some(18.0), i: Some(17.8), z: Some(17.5)
        });
        assert_eq!(stars[2], Star { // Optional fields should be None
            obj_id: 3, ra: 150.2, dec: 2.2, u: None, g: None, r: None, i: None, z: None
        });
    }

    #[test]
    fn test_load_stars_malformed_data() {
        // Malformed row: 'ra' is not a float
        let csv_content = "\u{FEFF}#Table1\n\
obj_id,ra,dec,u,g,r,i,z\n\
1,not_a_float,2.0,18.0,17.5,17.0,16.8,16.5\n\
2,150.1,2.1,19.0,18.5,18.0,17.8,17.5\n";

        let temp_file = create_temp_csv(csv_content).expect("Failed to create temp CSV for test_load_stars_malformed_data");
        let result = load_stars_from_csv(temp_file.path());

        assert!(result.is_err(), "Expected an error for malformed data, but got Ok");
        // Optionally, check the specific error type or message if important
        // e.g., result.unwrap_err().to_string().contains("invalid float literal")
    }
    
    #[test]
    fn test_load_stars_missing_required_field() {
        // Malformed row: 'ra' (required field) is missing.
        // csv crate by default might fill with None if Option, or fail if not Option.
        // For f64 (not Option<f64>), it should fail if a field is truly missing and not just empty.
        // If ra were Option<f64>, ``,`` would be None. But `ra: f64` requires a value.
        let csv_content = "\u{FEFF}#Table1\n\
obj_id,ra,dec,u,g,r,i,z\n\
1,,2.0,18.0,17.5,17.0,16.8,16.5\n"; // ra is empty

        let temp_file = create_temp_csv(csv_content).expect("Failed to create temp CSV for test_load_stars_missing_required_field");
        let result = load_stars_from_csv(temp_file.path());
        
        assert!(result.is_err(), "Expected an error for missing required field 'ra', but got Ok");
    }


    #[test]
    fn test_load_empty_data() {
        // CSV with only comment and header line, no data lines
        let csv_content = "\u{FEFF}#Table1\n\
obj_id,ra,dec,u,g,r,i,z\n";

        let temp_file = create_temp_csv(csv_content).expect("Failed to create temp CSV for test_load_empty_data");
        let result = load_stars_from_csv(temp_file.path());

        assert!(result.is_ok(), "load_stars_from_csv returned an error: {:?}", result.err());
        let stars = result.unwrap();
        assert!(stars.is_empty(), "Expected an empty vector of stars, but got {} stars", stars.len());
    }

    #[test]
    fn test_load_stars_file_not_found() {
        let result = load_stars_from_csv(Path::new("non_existent_file.csv"));
        assert!(result.is_err());
        // Check if the error is a file not found error (std::io::Error of kind NotFound)
        let error = result.unwrap_err();
        assert!(error.to_string().contains("No such file or directory") || error.to_string().contains("The system cannot find the file specified"));
    }
}