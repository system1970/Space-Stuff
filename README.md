# Nakshatra Space Stuff Projects

This repository contains multiple astronomy and data science tools for exploring and analyzing astronomical data, with a focus on the Sloan Digital Sky Survey (SDSS).

## Projects

### AstroQueryGPT

A RAG-powered, agentic Streamlit application for natural language querying of SDSS data.

- **Natural Language to SQL:** Ask astronomy questions in plain English and get SQL queries generated and executed automatically.
- **RAG Table Selection:** Uses semantic search to select the most relevant SDSS tables and fields for your query.
- **Agentic Correction Loop:** If a query fails, the agent retries with LLM-corrected SQL.
- **Data Verification:** Checks if the returned data is relevant and well-structured.
- **Interactive Visualization:** View results in a dataframe and as RA vs. Redshift plots (if available).
- **LLM Explanations:** Get plain-English explanations of every SQL query.
- **Full Agent Log:** Inspect every attempt, error, and correction.

See [`AstroQueryGPT/readme.md`](AstroQueryGPT/readme.md) for full details, setup, and usage instructions.

### star_index

A Rust-based command-line tool for fast nearest-neighbor search in astronomical catalogs (e.g., SDSS galaxies or stars).

- Uses R*-tree spatial indexing for efficient nearest-neighbor queries.
- Input: CSV files with RA/Dec and other columns.
- Output: Closest objects to a given sky position.

See [`star_index/readme.md`](star_index/readme.md) for usage and details.

## Requirements

- See each subproject's `requirements.txt` or `Cargo.toml` for dependencies.

## Attribution

- Code comments in this repository were generated using GPT-4.1.

## License

MIT License
