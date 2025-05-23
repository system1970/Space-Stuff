# Star Index: Nearest Neighbor Search for Astronomical Data

This project provides a fast command-line tool to find the nearest stars or galaxies in a large astronomical dataset using spatial indexing (R*-tree). The data used for demonstration is sourced from the `AstroQueryGPT` Streamlit app, specifically the `galaxies_that_are_elliptical.csv` file, which contains data on elliptical galaxies from the Sloan Digital Sky Survey (SDSS).

## Features

- Efficient nearest neighbor search using R*-tree spatial indexing.
- Command-line interface for flexible queries.
- Supports custom CSV datasets with RA/Dec coordinates.
- Outputs the closest objects to a given sky position.

## Data Source

The sample data (`galaxies_that_are_elliptical.csv`) was generated using the [AstroQueryGPT](../AstroQueryGPT/) Streamlit app. The app allows users to query the SDSS database using natural language, and the resulting data (e.g., elliptical galaxies) can be exported as CSV for use in this tool.

## Building the Project

To build the Star Index utility, you need to have Rust and Cargo installed. If you don't have them, please install them from [rust-lang.org](https://www.rust-lang.org/tools/install).

Once Rust and Cargo are set up, navigate to the `star_index` directory and run:

```bash
cargo build --release
```

This command compiles the project in release mode, creating an optimized executable located at `target/release/star_index` (or `target\release\star_index.exe` on Windows).

## Running the Utility

After building the project, you can run the utility from your terminal. The command structure is as follows:

```bash
# On Linux or macOS
./target/release/star_index --file <path_to_csv> --ra <ra_value> --dec <dec_value> --n <num_neighbors>

# On Windows
.\target\release\star_index.exe --file <path_to_csv> --ra <ra_value> --dec <dec_value> --n <num_neighbors>
```

**Command-line Arguments:**

-   `--file <path_to_csv>` or `-f <path_to_csv>`: **Required**. Path to the CSV file containing the astronomical data.
-   `--ra <ra_value>` or `-r <ra_value>`: **Required**. Right Ascension (RA) coordinate of the point to search from, in decimal degrees.
-   `--dec <dec_value>` or `-d <dec_value>`: **Required**. Declination (Dec) coordinate of the point to search from, in decimal degrees.
-   `--n <num_neighbors>` or `-n <num_neighbors>`: Optional. The number of nearest neighbors to find. Defaults to `5`.

**Example Usage:**

```bash
./target/release/star_index --file galaxies_that_are_elliptical.csv --ra 150.0 --dec 2.0 --n 3
```

This command will search for the 3 nearest neighbors to the sky coordinates RA=150.0 degrees, Dec=2.0 degrees, using data from the `galaxies_that_are_elliptical.csv` file.

## Expected CSV Format

The utility expects the input CSV file to adhere to a specific format:

1.  **First Line (Comment/BOM)**: The very first line of the CSV file is skipped. This line typically contains a Byte Order Mark (BOM) and often a comment (e.g., `#Table1` as seen in CSVs from AstroQueryGPT).
2.  **Second Line (Headers)**: The second line of the CSV file is treated as the header row. These headers are used to map data to the internal `Star` struct.
3.  **Data Lines**: Subsequent lines contain the actual data for each star/object.

**Required Columns in Header (must match these names):**

-   `obj_id`: A unique identifier for the object (unsigned 64-bit integer).
-   `ra`: Right Ascension in decimal degrees (floating-point number).
-   `dec`: Declination in decimal degrees (floating-point number).

**Optional Columns in Header:**

The following photometric magnitude columns can be included. If present, they will be parsed; if absent or empty for a row, they will be treated as `None`.

-   `u`: Magnitude in the u-band (floating-point number).
-   `g`: Magnitude in the g-band (floating-point number).
-   `r`: Magnitude in the r-band (floating-point number).
-   `i`: Magnitude in the i-band (floating-point number).
-   `z`: Magnitude in the z-band (floating-point number).

Any additional columns in the CSV file are ignored by the utility.

**Example CSV Snippet:**

```csv
\uFEFF#Table1,,,
obj_id,ra,dec,u,g,r,i,z
1237648720693842134,0.000065,14.16954,22.82473,20.96336,19.50488,18.80872,18.31594
1237648720693842135,0.000109,-0.529098,20.03806,18.80633,18.15585,17.82358,17.56448
...
```

## Example Output

The utility will print information for each of the N nearest neighbors found:

```
Loading stars from: galaxies_that_are_elliptical.csv
Loaded 2000 stars.
Building R*-tree index...
R*-tree index built.
Searching for 3 nearest neighbors to RA: 150, Dec: 2...
Neighbor 1: obj_id: 1237657584942448656, RA: 149.89200, Dec: 2.12300, Distance_sq: 0.01501
Neighbor 2: obj_id: 1237660750333018120, RA: 150.01500, Dec: 2.04500, Distance_sq: 0.00225
Neighbor 3: obj_id: 1237657584942448789, RA: 149.95000, Dec: 1.98000, Distance_sq: 0.00290
```
(Note: `Distance_sq` is the squared Euclidean distance in RA-Dec space.)

## How it Works

-   The utility starts by parsing the command-line arguments.
-   It then calls `load_stars_from_csv` to read the specified CSV file, skipping the first line and using the second as headers to deserialize star data into `Star` structs.
-   The loaded `Star` objects are wrapped in `StarPoint` structs, which are then used to build an R*-tree. An R*-tree is a spatial index that allows for efficient querying of multi-dimensional data (in this case, 2D sky coordinates RA and Dec).
-   Once the R*-tree is built, the `nearest_neighbor_iter` method is used to find the `n` closest `StarPoint` objects to the target RA and Dec coordinates provided by the user.
-   Finally, information about these nearest neighbors is printed to the console.

## Requirements

-   Rust (edition 2021)
-   Cargo (Rust's package manager and build system)
-   Dependencies (managed by Cargo, defined in `Cargo.toml`):
    -   `csv` (for CSV parsing)
    -   `rstar` (for R*-tree implementation)
    -   `serde` (for deserialization from CSV)
    -   `clap` (for command-line argument parsing)
    -   `tempfile` (for tests)

## Credits

-   Data obtained using [AstroQueryGPT](../AstroQueryGPT/)
-   SDSS SkyServer data used by AstroQueryGPT.

## License

MIT License
