"""
Build an interstate highway county-to-county edge list from Census TIGER/Line shapefiles.

This script:
1. Loads county boundary polygons and primary road geometries from local shapefiles
2. Filters roads to interstate highways only
3. For each road SEGMENT, finds all counties it passes through
4. Creates edges between counties that share a road segment (= sequential adjacency)
5. Optionally builds k-hop and all-pairs edges from the sequential graph

Requirements:
    pip install geopandas shapely pandas pyproj networkx

Data (download before running):
    - County boundaries: https://www2.census.gov/geo/tiger/TIGER2022/COUNTY/tl_2022_us_county.zip
    - Primary roads:     https://www2.census.gov/geo/tiger/TIGER2022/PRIMARYROADS/tl_2022_us_primaryroads.zip

    Unzip both into a data/ directory (or pass paths via --counties-shp and --roads-shp).

Usage:
    python build_interstate_edges.py --counties-shp data/tl_2022_us_county.shp \\
                                      --roads-shp data/tl_2022_us_primaryroads.shp \\
                                      --output-dir output/

Approach:
    Each row in the TIGER primary roads shapefile is a single road segment (a LineString
    or MultiLineString spanning some portion of a highway). If a single segment intersects
    two or more counties, those counties are sequentially adjacent along that highway.
    This avoids the need for linear referencing or route-level dissolve/merge, which are
    fragile with TIGER data.
"""

import argparse
import logging
import re
from pathlib import Path

import geopandas as gpd
import networkx as nx
import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exclude non-CONUS states (Alaska, Hawaii, territories)
# ---------------------------------------------------------------------------
NON_CONUS_STATE_FIPS = {"02", "15", "60", "66", "69", "72", "78"}
counties_shp = "data/Raw/Interstates_raw/tl_2022_us_county/tl_2022_us_county.shp"
roads_shp = "data/Raw/Interstates_raw/tl_2022_us_primaryroads/tl_2022_us_primaryroads.shp"

def load_counties(path: str, crs: str) -> gpd.GeoDataFrame:
    """Load county polygons, filter to CONUS, build 5-digit FIPS."""
    logger.info(f"Loading county boundaries from {path}")
    counties = gpd.read_file(path)
    counties["FIPS"] = counties["STATEFP"] + counties["COUNTYFP"]

    n_before = len(counties)
    counties = counties[~counties["STATEFP"].isin(NON_CONUS_STATE_FIPS)].copy()
    logger.info(f"Filtered to CONUS: {n_before} -> {len(counties)} counties")

    counties = counties.to_crs(crs)

    # Build a spatial index for fast intersection queries
    counties.sindex  # triggers index creation

    return counties[["FIPS", "STATEFP", "COUNTYFP", "NAME", "geometry"]]


def load_interstates(path: str, crs: str) -> gpd.GeoDataFrame:
    """Load primary roads, filter to interstates, extract route numbers."""
    logger.info(f"Loading primary roads from {path}")
    roads = gpd.read_file(path)

    if "RTTYP" in roads.columns:
        interstates = roads[roads["RTTYP"] == "I"].copy()
    else:
        logger.warning(
            "RTTYP column not found; falling back to FULLNAME pattern matching"
        )
        interstates = roads[
            roads["FULLNAME"].str.contains(r"I-\s*\d+", na=False)
        ].copy()

    logger.info(f"Found {len(interstates)} interstate road segments")

    def extract_route(name):
        if pd.isna(name):
            return None
        match = re.search(r"I-\s*(\d+)", name)
        return f"I-{match.group(1)}" if match else None

    interstates["route"] = interstates["FULLNAME"].apply(extract_route)
    interstates = interstates.dropna(subset=["route"])

    interstates = interstates.to_crs(crs)

    logger.info(
        f"Retained {len(interstates)} segments across "
        f"{interstates['route'].nunique()} unique routes"
    )
    return interstates[["route", "FULLNAME", "LINEARID", "geometry"]]


def _order_counties_along_segment(
    segment_geom, fips_list: list, counties: gpd.GeoDataFrame
) -> list:
    """
    Given a road segment geometry and a list of FIPS codes for counties
    it intersects, return the FIPS codes ordered by position along the segment.

    For each county, we compute the centroid of the intersection between
    the county polygon and the segment, then project that centroid onto
    the segment to get a linear reference position. Sorting by that
    position gives the correct sequential order.
    """
    county_rows = counties[counties["FIPS"].isin(fips_list)]
    positions = []

    for _, row in county_rows.iterrows():
        try:
            clipped = row.geometry.intersection(segment_geom)
            if clipped.is_empty:
                continue
            pt = clipped.centroid
            pos = segment_geom.project(pt)
            positions.append((row["FIPS"], pos))
        except Exception:
            continue

    positions.sort(key=lambda x: x[1])
    return [fips for fips, _ in positions]


def build_sequential_edges(
    interstates: gpd.GeoDataFrame, counties: gpd.GeoDataFrame
) -> pd.DataFrame:
    """
    For each road segment, find all counties it intersects.
    If a segment touches exactly 2 counties, create an edge between them.
    If a segment touches 3+ counties, order them along the segment by
    linear reference position and connect only consecutive pairs.
    """
    logger.info("Building sequential edges from road segments...")

    # Spatial join: for each road segment, find intersecting counties
    joined = gpd.sjoin(interstates, counties, how="inner", predicate="intersects")

    # joined has one row per (segment, county) pair
    # Group by segment (LINEARID) and collect the counties + route
    segment_counties = (
        joined.groupby("LINEARID")
        .agg(
            counties=("FIPS", list),
            route=("route", "first"),
        )
        .reset_index()
    )

    # Only keep segments that touch 2+ counties (border-crossing segments)
    multi = segment_counties[segment_counties["counties"].apply(len) >= 2]
    logger.info(
        f"{len(multi)} segments cross county boundaries "
        f"(out of {len(segment_counties)} total)"
    )

    # Build a lookup from LINEARID to segment geometry for ordering
    segment_geom_lookup = interstates.set_index("LINEARID")["geometry"].to_dict()

    n_two = 0
    n_three_plus = 0

    # Build edges from each multi-county segment
    edges = []
    for _, row in multi.iterrows():
        fips_list = sorted(set(row["counties"]))  # deduplicate within segment
        route = row["route"]

        if len(fips_list) == 2:
            # Simple case: exactly 2 counties, connect them directly
            n_two += 1
            edges.append(
                {"fips_i": fips_list[0], "fips_j": fips_list[1], "route": route}
            )
        else:
            # 3+ counties: order along the segment, connect consecutive only
            n_three_plus += 1
            segment_geom = segment_geom_lookup.get(row["LINEARID"])
            if segment_geom is None:
                # Fallback: skip this segment
                continue

            ordered = _order_counties_along_segment(
                segment_geom, fips_list, counties
            )
            for i in range(len(ordered) - 1):
                edges.append(
                    {"fips_i": ordered[i], "fips_j": ordered[i + 1], "route": route}
                )

    logger.info(
        f"Processed {n_two} two-county segments and "
        f"{n_three_plus} three-plus-county segments"
    )

    edges_df = pd.DataFrame(edges)

    if edges_df.empty:
        logger.warning("No sequential edges found!")
        return edges_df

    # Deduplicate: same county pair may appear from multiple segments
    # on the same or different routes
    edges_df["fips_lo"] = edges_df[["fips_i", "fips_j"]].min(axis=1)
    edges_df["fips_hi"] = edges_df[["fips_i", "fips_j"]].max(axis=1)

    deduped = (
        edges_df.groupby(["fips_lo", "fips_hi"])
        .agg(
            routes=("route", lambda x: ",".join(sorted(set(x)))),
            n_routes=("route", "nunique"),
        )
        .reset_index()
    )
    deduped = deduped.rename(columns={"fips_lo": "fips_i", "fips_hi": "fips_j"})
    deduped["edge_type"] = "sequential"

    return deduped


def build_khop_edges(sequential_edges: pd.DataFrame, k: int = 2) -> pd.DataFrame:
    """
    From the sequential edge list, build a graph and create edges
    between counties within k hops along any single interstate route.
    """
    if sequential_edges.empty:
        return pd.DataFrame()

    logger.info(f"Building {k}-hop edges from sequential graph...")

    # Build per-route graphs so hops are counted along individual routes
    all_edges = []

    # Expand routes (some edges have multiple routes)
    expanded = []
    for _, row in sequential_edges.iterrows():
        for route in row["routes"].split(","):
            expanded.append(
                {"fips_i": row["fips_i"], "fips_j": row["fips_j"], "route": route}
            )
    expanded_df = pd.DataFrame(expanded)

    for route, group in expanded_df.groupby("route"):
        G = nx.Graph()
        for _, row in group.iterrows():
            G.add_edge(row["fips_i"], row["fips_j"])

        # For each pair of nodes, check if shortest path <= k
        for source in G.nodes():
            lengths = nx.single_source_shortest_path_length(G, source, cutoff=k)
            for target, dist in lengths.items():
                if dist > 1 and source < target:  # skip direct neighbors and self
                    all_edges.append(
                        {
                            "fips_i": source,
                            "fips_j": target,
                            "route": route,
                            "hop_distance": dist,
                        }
                    )

    if not all_edges:
        return pd.DataFrame()

    edges_df = pd.DataFrame(all_edges)

    # Deduplicate across routes
    edges_df["fips_lo"] = edges_df[["fips_i", "fips_j"]].min(axis=1)
    edges_df["fips_hi"] = edges_df[["fips_i", "fips_j"]].max(axis=1)

    deduped = (
        edges_df.groupby(["fips_lo", "fips_hi"])
        .agg(
            routes=("route", lambda x: ",".join(sorted(set(x)))),
            n_routes=("route", "nunique"),
            min_hop_distance=("hop_distance", "min"),
        )
        .reset_index()
    )
    deduped = deduped.rename(columns={"fips_lo": "fips_i", "fips_hi": "fips_j"})
    deduped["edge_type"] = f"{k}hop"

    return deduped


def build_allpairs_edges(sequential_edges: pd.DataFrame) -> pd.DataFrame:
    """
    From the sequential edge list, build per-route graphs and create
    edges between all pairs of counties on the same route.
    """
    if sequential_edges.empty:
        return pd.DataFrame()

    logger.info("Building all-pairs edges from sequential graph...")

    expanded = []
    for _, row in sequential_edges.iterrows():
        for route in row["routes"].split(","):
            expanded.append(
                {"fips_i": row["fips_i"], "fips_j": row["fips_j"], "route": route}
            )
    expanded_df = pd.DataFrame(expanded)

    all_edges = []
    for route, group in expanded_df.groupby("route"):
        G = nx.Graph()
        for _, row in group.iterrows():
            G.add_edge(row["fips_i"], row["fips_j"])

        # For each connected component, create all-pairs edges
        for component in nx.connected_components(G):
            nodes = sorted(component)
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    # Compute hop distance along route
                    try:
                        dist = nx.shortest_path_length(G, nodes[i], nodes[j])
                    except nx.NetworkXNoPath:
                        dist = -1
                    all_edges.append(
                        {
                            "fips_i": nodes[i],
                            "fips_j": nodes[j],
                            "route": route,
                            "hop_distance": dist,
                        }
                    )

    if not all_edges:
        return pd.DataFrame()

    edges_df = pd.DataFrame(all_edges)

    edges_df["fips_lo"] = edges_df[["fips_i", "fips_j"]].min(axis=1)
    edges_df["fips_hi"] = edges_df[["fips_i", "fips_j"]].max(axis=1)

    deduped = (
        edges_df.groupby(["fips_lo", "fips_hi"])
        .agg(
            routes=("route", lambda x: ",".join(sorted(set(x)))),
            n_routes=("route", "nunique"),
            min_hop_distance=("hop_distance", "min"),
        )
        .reset_index()
    )
    deduped = deduped.rename(columns={"fips_lo": "fips_i", "fips_hi": "fips_j"})
    deduped["edge_type"] = "allpairs"

    return deduped


def build_route_county_lookup(
    interstates: gpd.GeoDataFrame, counties: gpd.GeoDataFrame
) -> pd.DataFrame:
    """Build a route-to-county lookup table for inspection."""
    joined = gpd.sjoin(interstates, counties, how="inner", predicate="intersects")

    lookup = (
        joined.groupby(["route", "FIPS"])
        .agg(county_name=("NAME", "first"), state_fips=("STATEFP", "first"))
        .reset_index()
        .sort_values(["route", "state_fips", "FIPS"])
    )
    return lookup


def summarize_edges(edges: pd.DataFrame, counties: gpd.GeoDataFrame, label: str):
    """Print summary statistics about the edge set."""
    n_edges = len(edges)
    if n_edges == 0:
        logger.info(f"[{label}] No edges generated.")
        return

    all_fips = set(edges["fips_i"]).union(set(edges["fips_j"]))
    n_counties = len(all_fips)
    total_counties = len(counties)

    logger.info(
        f"[{label}] {n_edges} edges connecting {n_counties} counties "
        f"({n_counties}/{total_counties} = {100*n_counties/total_counties:.1f}% of CONUS)"
    )

    if "n_routes" in edges.columns:
        multi_route = edges[edges["n_routes"] > 1]
        logger.info(
            f"[{label}] {len(multi_route)} edges shared by multiple interstates"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Build interstate highway county-to-county edge lists"
    )
    # parser.add_argument(
    #     "--counties-shp",
    #     required=True,
    #     help="Path to county boundaries shapefile (tl_2022_us_county.shp)",
    # )
    # parser.add_argument(
    #     "--roads-shp",
    #     required=True,
    #     help="Path to primary roads shapefile (tl_2022_us_primaryroads.shp)",
    # )
    # parser.add_argument(
    #     "--output-dir",
    #     default="output",
    #     help="Directory for output CSV files (default: output/)",
    # )
    parser.add_argument(
        "--mode",
        choices=["sequential", "allpairs", "khop", "all"],
        default="all",
        help="Edge construction mode (default: all)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=2,
        help="Number of hops for k-hop mode (default: 2)",
    )
    parser.add_argument(
        "--crs",
        default="EPSG:5070",
        help="Projected CRS for spatial operations (default: EPSG:5070, Conus Albers)",
    )
    args = parser.parse_args()
    
    output_dir = Path("data/Processed/Interstates")
    # output_dir = Path(args.output_dir)
    # output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load and prepare data
    # ------------------------------------------------------------------
    counties = load_counties(counties_shp, args.crs)
    interstates = load_interstates(roads_shp, args.crs)

    # ------------------------------------------------------------------
    # 2. Build sequential edges (foundation for all other modes)
    # ------------------------------------------------------------------
    seq_edges = build_sequential_edges(interstates, counties)
    summarize_edges(seq_edges, counties, "sequential")

    if args.mode in ("sequential", "all"):
        seq_path = output_dir / "interstate_edges_sequential.csv"
        seq_edges.to_csv(seq_path, index=False)
        logger.info(f"Sequential edges saved to {seq_path}")

    # ------------------------------------------------------------------
    # 3. Build derived edge sets from the sequential graph
    # ------------------------------------------------------------------
    if args.mode in ("khop", "all") and not seq_edges.empty:
        kh_edges = build_khop_edges(seq_edges, k=args.k)
        summarize_edges(kh_edges, counties, f"{args.k}-hop")
        kh_path = output_dir / f"interstate_edges_{args.k}hop.csv"
        kh_edges.to_csv(kh_path, index=False)
        logger.info(f"{args.k}-hop edges saved to {kh_path}")

    if args.mode in ("allpairs", "all") and not seq_edges.empty:
        ap_edges = build_allpairs_edges(seq_edges)
        summarize_edges(ap_edges, counties, "allpairs")
        ap_path = output_dir / "interstate_edges_allpairs.csv"
        ap_edges.to_csv(ap_path, index=False)
        logger.info(f"All-pairs edges saved to {ap_path}")

    # ------------------------------------------------------------------
    # 4. Route-county lookup table for inspection
    # ------------------------------------------------------------------
    lookup = build_route_county_lookup(interstates, counties)
    lookup_path = output_dir / "route_county_lookup.csv"
    lookup.to_csv(lookup_path, index=False)
    logger.info(f"Route-county lookup saved to {lookup_path}")

    logger.info("Done.")


if __name__ == "__main__":
    main()