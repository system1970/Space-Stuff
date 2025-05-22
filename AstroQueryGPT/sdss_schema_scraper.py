import asyncio
import json
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError

# --- Configuration ---
# URL for a specific table (e.g., apogeeField)
# You'll loop through your table names to generate these URLs
# TABLE_URL_TEMPLATE = "https://skyserver.sdss.org/dr16/en/help/browser/browser.aspx#&&history=description+{table_name}+U"

# Data for table names and their primary descriptions (same as before)
TABLES_TSV_DATA = """
name	description
apogeeDesign	Contains the plate design information for APOGEE plates.
apogeeField	Contains the basic information for an APOGEE field.
apogeeObject	Contains the targeting information for an APOGEE object.
apogeePlate	Contains all the information associated with an APOGEE plate.
apogeeStar	Contains data for an APOGEE star combined spectrum.
apogeeStarAllVisit	Links an APOGEE combined star spectrum with all visits for that star.
apogeeStarVisit	Links an APOGEE combined star spectrum with the visits used to create it.
apogeeVisit	Contains data for a particular APOGEE spectrum visit.
aspcapStar	Contains data for an APOGEE star ASPCAP entry.
aspcapStarCovar	Contains the covariance information for an APOGEE star ASPCAP entry.
AtlasOutline	Contains a record describing each AtlasOutline object
cannonStar	Contains the stellar parameters obtained from the Cannon.
DataConstants	The table stores the values of various enumerated and bitmask columns.
DBColumns	Every column of every table has a description in this table
DBObjects	Every SkyServer database object has a one line description in this table
DBViewCols	The columns of each view are stored for the auto-documentation
Dependency	Contains the detailed inventory of database objects
detectionIndex	Full list of all detections, with associated 'thing' assignment.
emissionLinesPort	Emission line kinematics results for SDSS and BOSS galaxies using GANDALF
Field	All the measured parameters and calibrations of a photometric field
FieldProfile	The mean PSF profile for the field as determined from bright stars.
FIRST	SDSS objects that match to FIRST objects have their match parameters stored here
Frame	Contains JPEG images of fields at various zoom factors, and their astrometry.
galSpecExtra	Estimated physical parameters for all galaxies in the MPA-JHU spectroscopic catalogue.
galSpecIndx	Index measurements of spectra from the MPA-JHU spectroscopic catalogue.
galSpecInfo	General information for the MPA-JHU spectroscopic re-analysis
galSpecLine	Emission line measurements from MPA-JHU spectroscopic reanalysis
HalfSpace	The constraints for boundaries of the the different regions
History	Contains the detailed history of schema changes
Inventory	Contains the detailed inventory of database objects
LoadHistory	Tracks the loading history of the database
mangaAlfalfaDR15	LFALFA data for the currently public MaNGA sample
mangaDAPall	Final summary file of the MaNGA Data Analysis Pipeline (DAP).
mangaDRPall	Final summary file of the MaNGA Data Reduction Pipeline (DRP).
mangaFirefly	Contains the measured stellar population parameters for each MaNGA galaxy.
mangaGalaxyZoo	Galaxy Zoo classifications for all MaNGA target galaxies
mangaHIall	Catalogue of Observed MaNGA Targets under program AGBT16A_095
mangaHIbonus	Catalogue of bonus detections under program AGBT16A_095
mangaPipe3D	Data products of MaNGA cubes derived using Pipe3D.
mangatarget	MaNGA Target Catalog
marvelsStar	Contains data for a MARVELS star.
marvelsVelocityCurveUF1D	Contains data for a particular MARVELS velocity curve using UF1D technique.
Mask	Contains a record describing the each mask object
MaskedObject	Contains the objects inside a specific mask
mastar_goodstars	Summary file of MaNGA Stellar Libary.
mastar_goodvisits	Summary file of all visits of stars included in MaNGA Stellar Libary.
Neighbors	All PhotoObj pairs within 0.5 arcmins
nsatlas	NASA-Sloan Atlas catalog
PawlikMorph	Morphological parameters for all galaxies in MaNGA DR15
PhotoObjAll	The full photometric catalog quantities for SDSS imaging.
PhotoObjDR7	Contains the spatial cross-match between DR8 photoobj and DR7 photoobj.
PhotoPrimaryDR7	Contains the spatial cross-match between DR8 primaries and DR7 primaries.
PhotoProfile	The annulus-averaged flux profiles of SDSS photo objects
Photoz	The photometrically estimated redshifts for all objects in the GalaxyTag view.
PhotozErrorMap	The error map of the photometric redshift estimation
Plate2Target	Which objects are in the coverage area of which plates?
PlateX	Contains data from a given plate used for spectroscopic observations.
ProfileDefs	This table contains the radii for the Profiles table
ProperMotions	Proper motions combining SDSS and recalibrated USNO-B astrometry.
qsoVarPTF	Variability information on eBOSS quasar targets using PTF lightcurves.
qsoVarStripe	Variability information on eBOSS quasar targets using SDSS stripe 82 data.
RC3	RC3 information for matches to SDSS photometry
RecentQueries	Record the ipAddr and timestamps of the last n queries
Region	Definition of the different regions
Region2Box	Tracks the parentage which regions contribute to which boxes
RegionArcs	Contains the arcs of a Region with their endpoints
RegionPatch	Defines the attributes of the patches of a given region
RegionTypes	This table stores the numeric codes and string names for the different Region types defined by convex spherical polygons.
Rmatrix	Contains various rotation matrices between spherical coordinate systems
ROSAT	ROSAT All-Sky Survey information for matches to SDSS photometry
Run	Contains the basic parameters associated with a run
RunShift	The table contains values of the various manual nu shifts for runs
sdssBestTarget2Sector	Map PhotoObj which are potential targets to sectors
SDSSConstants	This table contains most relevant survey constants and their physical units
sdssEbossFirefly	Contains the measured stellar population parameters for a spectrum.
sdssImagingHalfSpaces	Half-spaces (caps) describing the SDSS imaging geometry
sdssPolygon2Field	Matched list of polygons and fields
sdssPolygons	Polygons describing SDSS imaging data window function
sdssSector	Stores the information about set of unique Sector regions
sdssSector2Tile	Match tiles to sectors, wedges adn sectorlets, and vice versa.
sdssTargetParam	Contains the parameters used for a version of the target selection code
sdssTileAll	Contains information about each individual tile on the sky.
sdssTiledTargetAll	Information on all targets run through tiling for SDSS-I and SDSS-II
sdssTilingGeometry	Information about boundary and mask regions in SDSS-I and SDSS-II
sdssTilingInfo	Results of individual tiling runs for each tiled target
sdssTilingRun	Contains basic information for a run of tiling Contains basic information for a run of tiling
segueTargetAll	SEGUE-1 and SEGUE-2 target selection run on all imaging data
SiteConstants	Table holding site specific constants
SiteDBs	Table containing the list of DBs at this CAS site.
SiteDiagnostics	This table stores the full diagnostic snapshot after the last revision
SpecDR7	Contains the spatial cross-match between DR8 SpecObjAll and DR7 primaries.
SpecObjAll	Contains the measured parameters for a spectrum.
SpecPhotoAll	The combined spectro and photo parameters of an object in SpecObjAll
spiders_quasar	The SPIDERS quasar eRosita source
sppLines	Contains outputs from the SEGUE Stellar Parameter Pipeline (SSPP).
sppParams	Contains outputs from the SEGUE Stellar Parameter Pipeline (SSPP).
sppTargets	Derived quantities calculated by the SEGUE-2 target selection pipeline.
stellarMassFSPSGranEarlyDust	Estimated stellar masses for SDSS and BOSS galaxies (Granada method, early-star-formation with dust)
stellarMassFSPSGranEarlyNoDust	Estimated stellar masses for SDSS and BOSS galaxies (Granada method, early-star-formation with dust)
stellarMassFSPSGranWideDust	Estimated stellar masses for SDSS and BOSS galaxies (Granada method, early-star-formation with dust)
stellarMassFSPSGranWideNoDust	Estimated stellar masses for SDSS and BOSS galaxies (Granada method, early-star-formation with dust)
stellarMassPassivePort	Estimated stellar masses for SDSS and BOSS galaxies (Portsmouth method, passive model)
stellarMassPCAWiscBC03	Estimated stellar masses for SDSS and BOSS galaxies (Wisconsin method, Bruzual-Charlot models)
stellarMassPCAWiscM11	Estimated stellar masses for SDSS and BOSS galaxies (Wisconsin method, Maraston models)
stellarMassStarformingPort	Estimated stellar masses for SDSS and BOSS galaxies (Portsmouth method, star-forming model).
StripeDefs	This table contains the definitions of the survey layout as planned
Target	Keeps track of objects chosen by target selection and need to be tiled.
TargetInfo	Unique information for an object every time it is targeted
thingIndex	Full list of all 'things': unique objects in the SDSS imaging
TwoMass	2MASS point-source catalog quantities for matches to SDSS photometry
TwoMassXSC	2MASS extended-source catalog quantities for matches to SDSS photometry
USNO	SDSS objects that match to USNO-B objects have their match parameters stored here
Versions	Tracks the versioning history of the database
WISE_allsky	WISE All-Sky Data Release catalog
WISE_xmatch	Astrometric cross-matches between SDSS and WISE objects.
wiseForcedTarget	WISE forced-photometry of SDSS primary sources.
Zone	Table to organize objects into declination zones
zoo2MainPhotoz	Description: Morphological classifications of main-sample galaxies with photometric redshifts only from Galaxy Zoo 2
zoo2MainSpecz	Morphological classifications of main-sample spectroscopic galaxies from Galaxy Zoo 2.
zoo2Stripe82Coadd1	Morphological classifications of Stripe 82, coadded (sample 1) spectroscopic galaxies from Galaxy Zoo 2
zoo2Stripe82Coadd2	Morphological classifications of Stripe 82, coadded (sample 2) spectroscopic galaxies from Galaxy Zoo 2
zoo2Stripe82Normal	Morphological classifications of Stripe 82 normal-depth, spectroscopic galaxies from Galaxy Zoo 2
zooConfidence	Measures of classification confidence from Galaxy Zoo.
zooMirrorBias	Results from the bias study using mirrored images from Galaxy Zoo
zooMonochromeBias	Results from the bias study that introduced monochrome images in Galaxy Zoo.
zooNoSpec	Morphology classifications of galaxies without spectra from Galaxy Zoo
zooSpec	Morphological classifications of spectroscopic galaxies from Galaxy Zoo
zooVotes	Vote breakdown in Galaxy Zoo results.
""".strip() # Same as before
OUTPUT_SCHEMA_FILE = "sdss_schema_playwright_bs.json"
REQUEST_DELAY_S = 1.0 # Delay between processing each table

def parse_table_list_from_tsv(tsv_data: str) -> list[tuple[str, str]]:
    # ... (this function remains unchanged from your previous versions) ...
    parsed_tables = []
    lines = tsv_data.strip().split('\n')
    if not lines: return []
    header = [h.strip() for h in lines[0].split('\t')]
    if header != ['name', 'description']: print("Warning: TSV header mismatch")
    for line in lines[1:]:
        parts = line.split('\t')
        if len(parts) == 2:
            name, description = parts[0].strip(), parts[1].strip()
            if name and description: parsed_tables.append((name, description))
            elif name: parsed_tables.append((name, f"Schema for table {name}."))
        else: print(f"Warning: Skipping malformed TSV line: '{line}'")
    return parsed_tables

def parse_sdss_table_with_bs(html_content: str, table_name_for_debug: str) -> list[dict]:
    """
    Parses the SDSS schema table from the given HTML content using BeautifulSoup.
    """
    if not html_content:
        print(f"BS Parser ({table_name_for_debug}): No HTML content received.")
        return []

    soup = BeautifulSoup(html_content, "lxml") # Using lxml parser

    # Selector for the main schema table
    # <table border="0" bgcolor="#888888" width="720" ...>
    schema_table = soup.find("table", {
        "border": "0",
        "bgcolor": "#888888",
        "width": "720"
    })

    if not schema_table:
        print(f"BS Parser ({table_name_for_debug}): Target schema table not found.")
        # For debugging, save the HTML that BS tried to parse:
        # with open(f"debug_bs_input_{table_name_for_debug}.html", "w", encoding="utf-8") as f_html:
        # f_html.write(html_content)
        # print(f"BS Parser ({table_name_for_debug}): HTML content saved for inspection.")
        return []

    fields = []
    header_row = schema_table.find("tr")
    if not header_row:
        print(f"BS Parser ({table_name_for_debug}): No header row (tr) found in the table.")
        return []
        
    headers_html = header_row.find_all("td", class_="h")
    header_names = [th.get_text(strip=True).lower() for th in headers_html]
    
    column_map = {
        "name": "name", "type": "type", "length": "length", 
        "unit": "unit", "ucd": "ucd", "description": "description"
    }
    
    col_indices = {}
    for desired_key, html_header_name in column_map.items():
        try:
            col_indices[desired_key] = header_names.index(html_header_name)
        except ValueError:
            if desired_key in ["name", "type", "description"]:
                print(f"BS Parser ({table_name_for_debug}): Critical header '{html_header_name}' not found. Headers: {header_names}")
                return []
            col_indices[desired_key] = -1

    data_rows = schema_table.find_all("tr")[1:]

    for row_idx, row in enumerate(data_rows):
        cells = row.find_all("td", class_="v")
        if not cells: continue

        max_idx_needed = 0
        for key in col_indices:
            idx = col_indices[key]
            if idx != -1 and idx > max_idx_needed: max_idx_needed = idx
        
        if len(cells) <= max_idx_needed:
            # print(f"BS Parser ({table_name_for_debug}): Row {row_idx} has fewer cells ({len(cells)}) than needed ({max_idx_needed+1}).")
            continue

        field_data = {}
        valid_row = True
        for key, html_col_index in col_indices.items():
            if html_col_index != -1:
                try:
                    field_data[key] = cells[html_col_index].get_text(strip=True).replace('\xa0', ' ')
                except IndexError:
                    print(f"BS Parser ({table_name_for_debug}): IndexError accessing cell {html_col_index} in row {row_idx} with {len(cells)} cells.")
                    valid_row = False
                    break # Stop processing this row
            else:
                field_data[key] = ""
        
        if not valid_row:
            continue

        if field_data.get("name") and field_data.get("type"):
            fields.append({
                "name": field_data.get("name"),
                "type": field_data.get("type"),
                "length": field_data.get("length", ""),
                "unit": field_data.get("unit", ""),
                "ucd": field_data.get("ucd", ""),
                "description": field_data.get("description", "")
            })
            
    return fields


async def fetch_table_schema_html(page, url: str, table_name: str) -> str | None:
    """
    Navigates to the URL and waits for the schema table to load, then returns its HTML.
    The SDSS Schema Browser loads content into an iframe.
    """
    print(f"  Navigating to: {url}")
    try:
        await page.goto(url, wait_until="networkidle", timeout=30000) # Wait for network to be idle
        # Alternative: wait_until="domcontentloaded" then specific waits

        # The content is inside an iframe, typically named 'description' or it's the main one.
        # First, try to locate the iframe.
        # If the table is directly in the main page after JS load, this iframe part might not be needed,
        # but for browser.aspx, it's highly likely.
        
        # Wait for the specific table element *within the correct frame*
        # The table we want has attributes: border="0" bgcolor="#888888" width="720"
        # A more specific selector for a cell *within* that table can also work.
        # e.g., a header cell: 'table[border="0"][bgcolor="#888888"] td.h'
        
        # SDSS browser.aspx loads content into an iframe, usually with id "description" or it's the primary content frame.
        # Let's try to find the frame and then the table within it.
        
        iframe_selector = "iframe#description" # Common ID for the content frame
        table_selector_in_iframe = 'table[border="0"][bgcolor="#888888"][width="720"]'
        
        # Wait for the iframe itself to be attached and potentially loaded
        try:
            await page.wait_for_selector(iframe_selector, timeout=15000, state="attached")
            frame = page.frame(name="description") # Or use frame_locator(iframe_selector).first
            if not frame:
                frame = page.frame_locator(iframe_selector).first # Try locator
            
            if frame:
                print(f"  Found iframe 'description' for {table_name}. Waiting for table inside frame...")
                # Wait for the table to be visible within the iframe
                await frame.wait_for_selector(table_selector_in_iframe, timeout=20000, state="visible")
                print(f"  Table found within iframe for {table_name}.")
                # Get content of the iframe
                # Sometimes frame.content() is more reliable after waits
                html_content = await frame.content()
                return html_content
            else:
                print(f"  Iframe 'description' not found for {table_name}. Checking main page content.")
                
        except PlaywrightTimeoutError:
            print(f"  Timeout waiting for iframe or table within iframe for {table_name}. Will try main page.")
        except Exception as e:
            print(f"  Error accessing iframe for {table_name}: {e}. Will try main page.")


        # Fallback: If iframe not found or table not in iframe, try main page (less likely for browser.aspx)
        print(f"  Checking main page for table for {table_name}...")
        await page.wait_for_selector(table_selector_in_iframe, timeout=20000, state="visible")
        print(f"  Table found on main page for {table_name}.")
        html_content = await page.content()
        return html_content

    except PlaywrightTimeoutError:
        print(f"  Timeout waiting for table elements to load for {table_name} at {url}.")
        # await page.screenshot(path=f"debug_timeout_{table_name}.png") # Helpful for debugging
        return None
    except Exception as e:
        print(f"  Error during Playwright navigation/waiting for {table_name}: {e}")
        return None


async def main():
    all_tables_data = parse_table_list_from_tsv(TABLES_TSV_DATA)
    if not all_tables_data:
        print("No table metadata parsed from TSV. Exiting.")
        return

    full_schema_data = []

    async with async_playwright() as p:
        # browser = await p.chromium.launch(headless=True) # Set headless=False to watch
        browser = await p.chromium.launch(headless=False, args=["--no-sandbox", "--disable-setuid-sandbox"]) # Common args for CI/Docker
        
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        )
        page = await context.new_page()

        for i, (table_name, table_description) in enumerate(all_tables_data):
            # if table_name not in ["apogeeField", "AtlasOutline"]: # For testing specific tables
            #     continue 
            print(f"[{i+1}/{len(all_tables_data)}] Processing table: {table_name}")
            
            url = f"https://skyserver.sdss.org/dr16/en/help/browser/browser.aspx#&&history=description+{table_name}+U"
            
            html_content = await fetch_table_schema_html(page, url, table_name)
            
            fields = []
            if html_content:
                fields = parse_sdss_table_with_bs(html_content, table_name)
            
            if fields:
                print(f"  Successfully extracted {len(fields)} fields for {table_name}.")
            else:
                print(f"  Warning: No fields extracted for {table_name}.")

            full_schema_data.append({
                "name": table_name,
                "description": table_description,
                "fields": fields
            })
            await asyncio.sleep(REQUEST_DELAY_S) # Be polite

        await browser.close()

    # Save the full schema to JSON
    try:
        with open(OUTPUT_SCHEMA_FILE, "w", encoding="utf-8") as f:
            json.dump(full_schema_data, f, indent=2)
        print(f"\nDone. Full schema saved to {OUTPUT_SCHEMA_FILE}")
    except IOError as e:
        print(f"Error writing schema to file: {e}")

if __name__ == "__main__":
    asyncio.run(main())