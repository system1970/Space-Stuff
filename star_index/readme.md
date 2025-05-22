# Star Index: Nearest Neighbor Search for Astronomical Data

This project provides a fast command-line tool to find the nearest stars or galaxies in a large astronomical dataset using spatial indexing (R*-tree). The data used for demonstration is sourced from the `AstroQueryGPT` Streamlit app, specifically the `galaxies_that_are_elliptical.csv` file, which contains data on elliptical galaxies from the Sloan Digital Sky Survey (SDSS).

## Features

- Efficient nearest neighbor search using R*-tree spatial indexing
- Command-line interface for flexible queries
- Supports custom CSV datasets with RA/Dec coordinates
- Outputs the closest objects to a given sky position

## Data Source

The sample data (`galaxies_that_are_elliptical.csv`) was generated using the [AstroQueryGPT](../AstroQueryGPT/) Streamlit app. The app allows users to query the SDSS database using natural language, and the resulting data (e.g., elliptical galaxies) can be exported as CSV for use in this tool.

## Usage

### 1. Build the Project

    cargo build --release

### 2. Run the Nearest Neighbor Search

    target\release\star_index --file galaxies_that_are_elliptical.csv --ra <RA> --dec <Dec> --n <N>

- `--file`: Path to the CSV file (e.g., `galaxies_that_are_elliptical.csv`)
- `--ra`: Right Ascension (RA) coordinate to search from (in degrees)
- `--dec`: Declination (Dec) coordinate to search from (in degrees)
- `--n`: Number of nearest neighbors to return (default: 5)

#### Example

    target\release\star_index --file galaxies_that_are_elliptical.csv --ra 150.0 --dec 2.0 --n 3

This will print the 3 nearest elliptical galaxies to the given sky position.

## CSV Format

The CSV file should have the following columns (at minimum):

- `obj_id`: Unique object identifier
- `ra`: Right Ascension (degrees)
- `dec`: Declination (degrees)
- `u`, `g`, `r`, `i`, `z`: (Optional) Photometric magnitudes

Additional columns are ignored by the tool.

## Example Output

    obj_id: 1237657584942448656, RA: 149.892, Dec: 2.123
    obj_id: 1237660750333018120, RA: 150.015, Dec: 2.045
    ...

## How it Works

- Loads the CSV data and parses each row as a `Star` struct
- Builds an R*-tree spatial index for fast nearest neighbor queries
- Given a target RA/Dec, finds the closest objects in the dataset

## Requirements

- Rust (edition 2021)
- [rstar](https://crates.io/crates/rstar), [serde](https://crates.io/crates/serde), [csv](https://crates.io/crates/csv), [clap](https://crates.io/crates/clap)

Install dependencies and build with Cargo:

    cargo build --release

## Credits

- Data obtained using [AstroQueryGPT](../AstroQueryGPT/)
- SDSS SkyServer

## License

MIT License
