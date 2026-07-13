# Tanager-1 Data Access & Sensor Characterization

**Date:** 2026-04-27

---

## Section 1: STAC Catalog Access

The Tanager open data lives at a **static STAC catalog** (NOT a STAC API):

- **Root URL:** `https://www.planet.com/data/stac/tanager-core-imagery/catalog.json`
- **Catalog ID:** `tanager-core-imagery-v2`
- **STAC Version:** 1.1.0
- **License:** CC BY 4.0

The catalog contains 9 child collections: GHG Plumes, Energy and Mining, Natural Lands, Agriculture, Coastal and Water Bodies, Urban, Snow and Ice, Fire, ROCX 2025.

**Critical distinction:** No authentication required. This is a static STAC catalog hosted on GCS. Planet's authenticated STAC API (`api.planet.com/x/data/`) does **NOT** list TanagerScene. Must use `pystac` (static catalog reader), **NOT** `pystac-client`.

### Python Workflow

```python
import pystac

catalog = pystac.Catalog.from_file(
    "https://www.planet.com/data/stac/tanager-core-imagery/catalog.json"
)
fire = catalog.get_child("fire")
for item in fire.get_items():
    print(item.id, item.datetime, item.bbox)
```

Also available via Google Earth Engine as a community dataset:
`projects/sat-io/open-datasets/PLANET/TANAGER_HYPERSPECTRAL`

---

## Section 2: Data Format & Structure

Files are in **HDF-EOS5 format** (`.h5`). Internal hierarchy:

```
/HDFEOS/SWATHS/HYP/
  Data_Fields/
    toa_radiance (or surface_reflectance)  # [bands, rows, cols] Float32
  Geolocation_Fields/
    Latitude, Longitude  # [rows, cols] Float64
  Metadata/
    wavelengths, fwhm, geotransform, EPSG
```

### Product Types

4 product types are available per scene:

| Product | Ortho? | Radiometric | Our Use |
|---------|--------|-------------|---------|
| Basic Radiance | No | TOA Radiance | Custom atm correction only |
| Ortho Radiance | Yes (UTM) | TOA Radiance | If radiance is needed |
| Basic Surface Reflectance | No | SR (ISOFIT v2.9.5) | Advanced analysis |
| **Ortho Surface Reflectance** | **Yes (UTM)** | **SR (ISOFIT v2.9.5)** | **Primary product** |

### File Naming Convention

`{YYYYMMDD}_{HHMMSS}_{hundredths}_{satellite_id}_{asset_type}.h5`

- Satellite ID `4001` = Tanager-1

### Band Metadata

Per-band metadata fields:

- `wavelengths`: center wavelength in nm (first band: 376.44 nm)
- `fwhm`: full-width half-maximum in nm (~5.4 nm)
- `applied_radiometric_coefficient`: calibration factor

### Scene Characteristics

- **Dimensions:** ~600 cols x 325–750 rows x 426 bands x Float32
- **File size:** ~480 MB per scene per product
- **Detector:** 640x480 MCT focal plane array, SNR 300–600

### Planet Processing Chain

Dark subtraction, pedestal correction, flat field, bad pixel correction, optical scatter correction, ghost correction, radiometric calibration, OSF seam correction, orthorectification (OneAtlas + DEM), and ISOFIT v2.9.5 atmospheric correction for SR products.

---

## Section 3: HyperCoast Integration

### `hypercoast.read_tanager()` Signature

```python
hypercoast.read_tanager(
    filepath,           # Local path or HTTPS URL
    bands=None,         # Optional band indices
    stac_url=None,      # Optional STAC item for wavelength metadata
    wavelengths=None,   # Direct wavelength values (nm)
    product=None,       # Force product type
    **kwargs
) -> xr.Dataset
```

Returns `xarray.Dataset` with dims `(wavelength, y, x)`. Coordinates include `wavelength` (nm), `fwhm` (nm), `latitude`, and `longitude`. Auto-discovers HDF5 layout via `_discover_tanager_layout()`.

### Additional HyperCoast Functions

| Function | Purpose |
|----------|---------|
| `grid_tanager()` | Regrid basic (non-ortho) products |
| `filter_tanager()` | Spatial subset |
| `extract_tanager()` | Point extraction |
| `Map.add_tanager()` | Interactive visualization |
| `image_cube()` | 3D rendering |

### Division of Responsibility

**What HyperCoast handles:** All 4 product I/O, wavelength-aware xarray, visualization, spectral extraction.

**What we build on top:** MESMA unmixing (SPy/mesma package), spectral indices, band math, continuum removal, multi-temporal analysis, export to COG/GeoPackage.

---

## Section 4: Available Fire Scenes

The fire collection contains 11–12 scenes spanning December 2024 through September 2025.

| Date | Scene ID | Phase | Time Since Ignition |
|------|----------|-------|---------------------|
| 2024-12-15 | 20241215_185916_33_4001 | **Pre-fire baseline** | -23 days |
| 2025-01-23 | 20250123_185507_64_4001 | **Immediate post-fire** | +16 days |
| 2025-01-23 | 20250123_185518_92_4001 | Post-fire (adjacent swath) | +16 days |
| 2025-04-07 | 20250407_192235_24_4001 | **Early recovery** | +90 days |
| 2025-04-07 | 20250407_192229_16_4001 | Early recovery (adjacent) | +90 days |
| 2025-07-24 | 20250724_190927_83_4001 | **Mid recovery** | ~200 days |
| 2025-07-26 | 20250726_192343_21_4001 | Mid recovery | ~200 days |
| 2025-07-26 | 20250726_192422_87_4001 | Mid recovery (adjacent) | ~200 days |
| 2025-09-02 | 20250902_190116_02_4001 | **Late recovery** | ~240 days |
| 2025-09-02 | 20250902_190121_86_4001 | Late recovery (adjacent) | ~240 days |
| 2025-09-20 | 20250920_193207_61_4001 | Likely Northern Arizona | N/A |

### Spatial Extent

- **Bounding box:** lon -118.91 to -111.82, lat 33.90 to 38.76
- **Western scenes:** Los Angeles area — Palisades Fire (23,448 ac) + Eaton Fire (14,021 ac), both ignited January 7, 2025
- **Eastern scenes:** Northern Arizona

---

## Section 5: Sensor Comparison

| Sensor | Bands | Spacing | Range (nm) | GSD | Format | Auth Required | Python Tool |
|--------|-------|---------|------------|-----|--------|---------------|-------------|
| AVIRIS | 224 | 10 nm | 380–2500 | 4–20 m | ENVI/HDF5 | NASA Earthdata | spectral |
| EMIT | 285 | 7.4 nm | 380–2500 | 60 m | netCDF4 | NASA Earthdata | earthaccess |
| EnMAP | 224 | ~10 nm | 420–2450 | 30 m | GeoTIFF | DLR portal (manual) | EnMAP-Box |
| PRISMA | 239 | ~12 nm | 400–2505 | 30 m | HDF5 | ASI portal (manual) | prismaread (frozen) |
| **Tanager** | **426** | **~5 nm** | **380–2500** | **30 m** | **HDF-EOS5** | **None (open)** | **HyperCoast** |

Tanager has the lowest-friction access of all sensors and the highest spectral resolution in this comparison.

---

## Section 6: Storage & Compute Requirements

### Storage

| Data Set | Size |
|----------|------|
| Fire collection (ortho SR only, recommended) | ~6 GB |
| Fire collection (all 4 products) | ~24 GB |
| Full open data (all collections) | ~300 GB |
| Working copies + intermediates | 2–3x raw |

### Compute

| Operation | Time |
|-----------|------|
| Loading a scene | ~2–5 s |
| MESMA per scene | ~5–15 min |
| Full time series | ~1–2 hr |

- **RAM per loaded scene:** ~2 GB
- **GPU:** Not required
- **Recommended:** 16 GB RAM handles individual scenes; 32 GB recommended for full time series pipeline

---

## Section 7: Recommended Data Access Workflow

### Step 1: Catalog Discovery

```python
import pystac

catalog = pystac.Catalog.from_file(
    "https://www.planet.com/data/stac/tanager-core-imagery/catalog.json"
)
fire_collection = catalog.get_child("fire")
items = list(fire_collection.get_items())
print(f"Found {len(items)} fire scenes")
```

### Step 2: Scene Selection and Download

```python
import requests
from pathlib import Path

# No authentication required
def download_asset(item, asset_key, output_dir):
    asset = item.assets[asset_key]
    url = asset.href
    filename = Path(output_dir) / Path(url).name
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return filename
```

### Step 3: Load with HyperCoast

```python
import hypercoast

ds = hypercoast.read_tanager(
    filepath="/data/fire/20250123_185507_64_4001_ortho_surface_reflectance.h5"
)
# ds dims: (wavelength, y, x)
# ds.wavelength: center wavelengths in nm
```

### Step 4: Spectral Analysis

```python
import numpy as np
from spectral import mesma  # SPy

# Convert xarray to numpy for SPy
data_array = ds["surface_reflectance"].values  # (wavelength, y, x)
wavelengths = ds.wavelength.values

# Transpose to (y, x, wavelength) for SPy
data_spy = np.transpose(data_array, (1, 2, 0))
```

**Key recommendation:** Download only `ortho_surface_reflectance` for fire scenes initially (~6 GB). Add `basic_radiance` only if custom atmospheric correction is needed.

---

## Section 8: Open Questions

1. **Asset key naming:** Exact asset keys in STAC items need to be confirmed by inspecting a live `item.json` — naming conventions may differ from documentation.
2. **SR product availability:** Do all 12 fire scenes have SR products, or only radiance? SR processing via ISOFIT was added later in Planet's pipeline; earlier scenes may be radiance-only.
3. **Pre/post spatial overlap:** What is the actual spatial overlap between the Dec 15, 2024 pre-fire scene and the Jan 23, 2025 post-fire scenes? This determines whether a true pre/post comparison is possible.
4. **Cross-sensor validation:** Are EMIT scenes available over the same LA area and time period for cross-sensor validation of spectral indices and burn severity estimates?

---

## References

- Planet Tanager Documentation: https://docs.planet.com/data/imagery/tanager/
- Planet STAC Catalog: https://www.planet.com/data/stac/tanager-core-imagery/catalog.json
- HyperCoast `tanager.py` source: https://github.com/opengeos/HyperCoast/blob/main/hypercoast/tanager.py
- HyperCoast Tanager Example: https://hypercoast.org/examples/tanager/
- Planet Open Data Announcement: https://www.planet.com/pulse/unleash-the-power-of-hyperspectral-over-50-tanager-radiance-datasets-now-available-on-planet-s/
- TanagerScene Item Type Docs: https://docs.planet.com/data/imagery/tanager/item-types/tanagerscene/
- GEE Community Catalog: https://gee-community-catalog.org/projects/tanager/
- Independent Technical Analysis: https://tech.marksblogg.com/planet-labs-tanager-hyperspectral-satellite-images.html
