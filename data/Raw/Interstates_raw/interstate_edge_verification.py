"""
Validate interstate edges against the Census Bureau County Adjacency File.

Every sequential interstate edge should also be a geographic adjacency edge —
if county A and county B are consecutive along an interstate, they must share
a border. Any interstate edge that is NOT in the adjacency file is suspect.

Data:
    - Census County Adjacency File (pipe-delimited):
      https://www.census.gov/geographies/reference-files/time-series/geo/county-adjacency.html
      Download the text file (e.g., county_adjacency2025.txt)

Usage:
    python validate_interstate_edges.py \
        --interstate-edges output/interstate_edges_sequential.csv \
        --adjacency-file data/county_adjacency2025.txt
"""

# import argparse
import csv
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_adjacency_set(path: str) -> set:
    """
    Parse the Census County Adjacency File into a set of (fips_lo, fips_hi) tuples.

    The file is pipe-delimited with the layout:
        County GEOID | County Name | Neighbor GEOID | Neighbor Name

    Each adjacency pair appears twice (once in each direction).
    We normalize to (min, max) so each pair appears once.
    """
    logger.info(f"Loading adjacency file from {path}")
    adjacency_pairs = set()

    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f, delimiter="|")
        header = next(reader, None)  # skip header row

        current_geoid = None
        for row in reader:
            if len(row) < 5:
                continue

            # Columns: County Name, County GEOID, Neighbor Name, Neighbor GEOID, Length
            county_geoid = row[1].strip()
            neighbor_geoid = row[3].strip()

            # The file uses a "block" format: the county GEOID is only
            # listed on the first row of each block, subsequent rows in
            # the same block have it blank.
            if county_geoid:
                current_geoid = county_geoid

            if not current_geoid or not neighbor_geoid:
                continue

            # Skip self-adjacency
            if current_geoid == neighbor_geoid:
                continue

            pair = tuple(sorted([current_geoid, neighbor_geoid]))
            adjacency_pairs.add(pair)

    logger.info(f"Loaded {len(adjacency_pairs)} unique adjacency pairs")
    return adjacency_pairs


def load_interstate_edges(path: str) -> pd.DataFrame:
    """Load the interstate edge CSV."""
    logger.info(f"Loading interstate edges from {path}")
    edges = pd.read_csv(path, dtype=str)

    # Ensure FIPS codes are zero-padded to 5 digits
    edges["fips_i"] = edges["fips_i"].str.zfill(5)
    edges["fips_j"] = edges["fips_j"].str.zfill(5)

    logger.info(f"Loaded {len(edges)} interstate edges")
    return edges


def validate(interstate_edges: pd.DataFrame, adjacency_set: set) -> pd.DataFrame:
    """
    Check each interstate edge against the adjacency set.
    Returns a DataFrame with a 'valid' column indicating whether
    the edge is also a geographic adjacency.
    """
    results = []
    for _, row in interstate_edges.iterrows():
        pair = tuple(sorted([row["fips_i"], row["fips_j"]]))
        is_adjacent = pair in adjacency_set
        results.append(
            {
                "fips_i": row["fips_i"],
                "fips_j": row["fips_j"],
                "routes": row.get("routes", ""),
                "is_adjacent": is_adjacent,
            }
        )

    results_df = pd.DataFrame(results)

    n_valid = results_df["is_adjacent"].sum()
    n_invalid = len(results_df) - n_valid
    pct_valid = 100 * n_valid / len(results_df) if len(results_df) > 0 else 0

    logger.info(f"Validation results:")
    logger.info(f"  {n_valid}/{len(results_df)} edges are valid adjacencies ({pct_valid:.1f}%)")
    logger.info(f"  {n_invalid} edges are NOT in the adjacency file (suspect)")

    return results_df


def main():
    # parser = argparse.ArgumentParser(
    #     description="Validate interstate edges against Census county adjacency"
    # )
    # # parser.add_argument(
    # #     "--interstate-edges",
    # #     required=True,
    # #     help="Path to interstate_edges_sequential.csv",
    # # )
    # # parser.add_argument(
    # #     "--adjacency-file",
    # #     required=True,
    # #     help="Path to Census county adjacency file (pipe-delimited .txt)",
    # # )
    # parser.add_argument(
    #     "--output",
    #     default=None,
    #     help="Path to save validation results CSV (default: print suspect edges only)",
    # )
    # args = parser.parse_args()

    adjacency_file = "data/Raw/Interstates_raw/county_adjacency2025.txt"
    interstate_edges = "data/Raw/Interstates_raw/interstate_edges_sequential.csv"
    output_dir = "data/Raw/Interstates_raw/interstate_edge_validation_results.csv"

    adjacency_set = load_adjacency_set(adjacency_file)
    interstate_edges = load_interstate_edges(interstate_edges)
    results = validate(interstate_edges, adjacency_set)

    # Show suspect edges
    suspect = results[~results["is_adjacent"]]
    if not suspect.empty:
        logger.info(f"\nSuspect edges (not in adjacency file):")
        for _, row in suspect.iterrows():
            logger.info(f"  {row['fips_i']} <-> {row['fips_j']}  routes: {row['routes']}")
    else:
        logger.info("All interstate edges are valid adjacencies.")

    # if args.output:
    results.to_csv(output_dir, index=False)
    logger.info(f"Full results saved to {output_dir}")


if __name__ == "__main__":
    main()