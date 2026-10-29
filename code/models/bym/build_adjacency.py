"""Build the admin1 adjacency graph used by the BYM2 model.

For each country, read the admin1 shapefile, compute queen contiguity
(two regions are neighbours if their boundaries share at least one point),
and match the shapefile names to the `adm1_name` used in the weekly data.

Outputs, written next to the weekly data:
    data/new/<country>/<country>-graph-nodes.csv
        node_id (1-based, Stan index), adm1_name (data naming), in_data
    data/new/<country>/<country>-graph-edges.csv
        node1, node2 (1-based, node1 < node2), adm1_name_1, adm1_name_2

Every region of the shapefile is a node, including regions that never enter
the likelihood (Nigerian states without 90 day surveys, Tibesti in Chad).
Keeping them preserves the contiguity structure of the ICAR prior.

Usage (from the project root):
    python code/models/bym/build_adjacency.py
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd
from libpysal.weights import Queen

ROOT = Path(__file__).resolve().parents[3]

# Shapefile name -> data name, for names that do not match after case folding.
# Chad shapefile names carry broken accents (e.g. "GUÃRA"), so Chad is matched
# on the pcode instead.
CONFIG = {
    "nigeria": {
        "shapefile": ROOT / "data/nigeria_geospatial/nigeria_1.shp",
        "shp_col": "state",
        "weekly": ROOT / "data/new/nigeria/nigeria-weekly-with-features.csv",
        "overrides": {"fct": "Abuja", "nasarawa": "Nassarawa"},
    },
    "chad": {
        "shapefile": ROOT / "data/chad_geospatial/tcd_admin1.shp",
        "shp_col": "adm1_pcode",
        "weekly": ROOT / "data/new/chad/chad-weekly-with-features.csv",
        "overrides": {
            "TCD04": "Guera",
            "TCD14": "Ouaddai",
            "TCD16": "Tandjile",
            "TCD18": "Ndjamzna",  # spelling used in the weekly data
            "TCD19": "Barh El Gazal",
            "TCD22": "Tibesti",  # not in the weekly data
        },
        "fallback_col": "adm1_name",
    },
}


def match_names(gdf, cfg, data_names):
    """Return the data-style adm1_name of every shapefile row."""
    lookup = {n.lower(): n for n in data_names}
    out = []
    for _, row in gdf.iterrows():
        key = row[cfg["shp_col"]]
        if key in cfg["overrides"]:
            out.append(cfg["overrides"][key])
            continue
        raw = row[cfg.get("fallback_col", cfg["shp_col"])]
        name = lookup.get(str(raw).lower())
        if name is None:
            raise ValueError(f"No match in the data for shapefile region '{raw}'")
        out.append(name)
    return out


def build(country):
    cfg = CONFIG[country]
    gdf = gpd.read_file(cfg["shapefile"])
    data_names = sorted(pd.read_csv(cfg["weekly"])["adm1_name"].unique())

    gdf["adm1_name"] = match_names(gdf, cfg, data_names)
    missing = set(data_names) - set(gdf["adm1_name"])
    if missing:
        raise ValueError(f"Regions of the data absent from the shapefile: {missing}")

    # Order nodes alphabetically, as the weekly data are sorted by adm1_name
    gdf = gdf.sort_values("adm1_name").reset_index(drop=True)

    w = Queen.from_dataframe(gdf, use_index=False)
    if w.islands:
        raise ValueError(f"Isolated regions: {gdf.loc[w.islands, 'adm1_name'].tolist()}")
    if w.n_components != 1:
        raise ValueError("The graph is not connected; the scaling factor must be computed per component")

    nodes = pd.DataFrame({
        "node_id": range(1, len(gdf) + 1),
        "adm1_name": gdf["adm1_name"],
        "in_data": gdf["adm1_name"].isin(data_names),
    })
    edges = sorted({(min(i, j) + 1, max(i, j) + 1) for i, nb in w.neighbors.items() for j in nb})
    edges = pd.DataFrame(edges, columns=["node1", "node2"])
    edges["adm1_name_1"] = nodes["adm1_name"].values[edges["node1"] - 1]
    edges["adm1_name_2"] = nodes["adm1_name"].values[edges["node2"] - 1]

    out_dir = cfg["weekly"].parent
    nodes.to_csv(out_dir / f"{country}-graph-nodes.csv", index=False)
    edges.to_csv(out_dir / f"{country}-graph-edges.csv", index=False)
    print(f"{country}: {len(nodes)} nodes ({(~nodes.in_data).sum()} without data), {len(edges)} edges")


if __name__ == "__main__":
    for country in CONFIG:
        build(country)
