# Zenodo Data Deposit Guide

Upload FireSpec's derived products as a citable Zenodo dataset. This earns tie-breaker
points for the competition and demonstrates open-data commitment.

## Step 1: Create a Zenodo account

Go to https://zenodo.org and sign in (GitHub SSO works).

## Step 2: Create a new upload

Click "New upload" on the Zenodo dashboard.

## Step 3: Upload these artifacts

Collect the following files into a single directory and upload them:

| File | Size | Description |
|------|------|-------------|
| `outputs/aviris3_validation/tanager_palisades_fractions.nc` | 24 MB | Tanager MESMA char/ash fractions over Palisades |
| `outputs/aviris3_validation/cross_validation_results.json` | 7 KB | Per-granule cross-sensor accuracy metrics |
| `data/reference/dins/palisades_dins.geojson` | 3.1 MB | CAL FIRE DINS structure-damage survey |
| `data/reference/baer/hughes_sbs.tif` | 194 KB | BAER Soil Burn Severity reference raster |
| `outputs/20241215_frac_char.tif` | 152 KB | Pre-fire MESMA char fraction |
| `outputs/20241215_frac_pv.tif` | 217 KB | Pre-fire MESMA PV fraction |
| `outputs/20241215_frac_npv.tif` | 178 KB | Pre-fire MESMA NPV fraction |
| `outputs/20241215_frac_soil.tif` | 129 KB | Pre-fire MESMA soil fraction |
| **Total** | **~28 MB** | |

## Step 4: Fill in the metadata

Use these values:

- **Title:** FireSpec: Derived Wildfire Products from Planet Tanager-1 Hyperspectral Imagery (2025 LA Fires)
- **Upload type:** Dataset
- **Description:**

> Derived data products from the FireSpec hyperspectral wildfire analysis toolkit,
> built for the Planet Tanager Open Data Competition. Products include MESMA spectral
> unmixing fraction maps (char, PV, NPV, soil), AVIRIS-3 cross-sensor validation
> results, CAL FIRE DINS structure-damage reference data, and BAER Soil Burn Severity
> ground truth for the January 2025 Palisades and Hughes fires in Los Angeles, CA.
> Source code and notebooks at https://github.com/gpriceless/tanager_wildfire

- **Creators:** Price, Gabriel
- **License:** MIT (for derived products); CC BY 4.0 (for DINS/BAER reference data)
- **Keywords:** hyperspectral, wildfire, burn severity, MESMA, Tanager-1, Planet, LFMC, spectral unmixing, Los Angeles fires
- **Related identifiers:**
  - `https://github.com/gpriceless/tanager_wildfire` (is supplemented by this upload)
  - `10.3334/ORNLDAAC/2357` (AVIRIS-3 source data)
  - `10.1038/s41597-024-03159-6` (Globe-LFMC 2.0)
- **Communities:** Search for "remote sensing" or "geospatial" communities

## Step 5: Publish

Click "Publish." Zenodo assigns a DOI immediately.

## Step 6: Update the README

Add the Zenodo DOI badge and link to the README:

```markdown
[![DOI](https://zenodo.org/badge/DOI/YOUR_DOI_HERE.svg)](https://doi.org/YOUR_DOI_HERE)
```

And add a "Data" section:

```markdown
## Data Products

Derived products (MESMA fractions, validation artifacts) are archived on Zenodo:
[DOI: YOUR_DOI_HERE](https://doi.org/YOUR_DOI_HERE)
```
