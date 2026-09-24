#!/usr/bin/env python3
"""Fetch official fire perimeters for the scenes this project analyses.

Source is the NIFC/WFIGS Interagency Perimeters feature service, which is the
authoritative published footprint for each incident. Incidents are matched on
name *plus* discovery date *plus* state — "Palisades" alone returns six fires
across three states and seven years — and the returned acreage is checked
against the published figure before anything is written.

These perimeters are the external referent for what the raster products show.
They are what confirmed that ``20241215_frac_char.tif`` resolves the Franklin
Fire and not the Palisades Fire.

Usage::

    python3 scripts/fetch_fire_perimeters.py
"""
from __future__ import annotations

import datetime
import json
import pathlib
import urllib.parse
import urllib.request

SERVICE = (
    "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/"
    "WFIGS_Interagency_Perimeters/FeatureServer/0/query"
)
OUT = pathlib.Path("data/reference/perimeters/la_fires_2025.geojson")

# (query name, discovery date, published acres, display name)
#
# Kenneth is here because unmixing the 2025-01-23 swath found a 1.4 km² patch of
# char at 34.174 N, -118.686 W that sits outside both of the other two
# perimeters. It is a real burn scar — Kenneth ignited on 2025-01-09, two weeks
# before the scene — and while it was missing from this file it was being
# counted as unburned reference area in every inside/outside comparison.
WANTED = [
    ("PALISADES", "2025-01-07", 23448, "Palisades"),
    ("Franklin", "2024-12-10", 4089, "Franklin"),
    ("KENNETH", "2025-01-09", 999, "Kenneth"),
]

ACRE_TOLERANCE = 0.02


def fetch(name: str) -> dict:
    query = {
        "where": f"poly_IncidentName='{name}' AND attr_POOState='US-CA'",
        "outFields": (
            "poly_IncidentName,attr_FireDiscoveryDateTime,poly_GISAcres,"
            "attr_POOState,attr_IncidentName"
        ),
        "returnGeometry": "true",
        "outSR": "4326",
        "f": "geojson",
    }
    url = SERVICE + "?" + urllib.parse.urlencode(query)
    with urllib.request.urlopen(url, timeout=120) as response:
        return json.load(response)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    features = []

    for name, want_date, want_acres, display in WANTED:
        collection = fetch(name)
        match = None
        for feature in collection.get("features", []):
            stamp = feature["properties"].get("attr_FireDiscoveryDateTime")
            if not stamp:
                continue
            found = datetime.datetime.fromtimestamp(
                stamp / 1000, datetime.UTC
            ).strftime("%Y-%m-%d")
            if found == want_date:
                match = feature
                break

        if match is None:
            raise SystemExit(
                f"No {name} perimeter with discovery date {want_date}. The service "
                "may have re-dated the incident; check before loosening the match."
            )

        acres = match["properties"]["poly_GISAcres"]
        drift = abs(acres - want_acres) / want_acres
        if drift > ACRE_TOLERANCE:
            raise SystemExit(
                f"{name}: service reports {acres:,.0f} acres, published figure is "
                f"{want_acres:,} ({drift:.1%} apart). Refusing to write a perimeter "
                "that disagrees with the record."
            )

        match["properties"]["fire"] = display
        features.append(match)
        print(f"  {display:10s} {want_date}  {acres:>9,.0f} acres  "
              f"{match['geometry']['type']}")

    OUT.write_text(json.dumps({
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
        "features": features,
    }))
    print(f"\nWrote {OUT} ({OUT.stat().st_size / 1000:.0f} kB)")


if __name__ == "__main__":
    print("Fetching NIFC/WFIGS perimeters...")
    main()
