"""Statistics computation for datasets."""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None


def compute_stats(
    dataset_path: Path,
    keys: Optional[List[str]] = None,
    output_file: Optional[Path] = None,
) -> Dict[str, Dict[str, List[float]]]:
    """Compute normalization statistics for dataset.

    Computes mean, std, min, max for observation.state and action columns.

    Args:
        dataset_path: Path to dataset.
        keys: Specific columns to compute stats for. Defaults to
              ["observation.state", "action"].
        output_file: If provided, saves stats to this file.

    Returns:
        Dictionary mapping column names to their statistics.
    """
    if pq is None:
        raise ImportError("pyarrow required: pip install pyarrow")

    dataset_path = Path(dataset_path)
    keys = keys or ["observation.state", "action"]

    # Find all parquet files
    parquet_files = sorted((dataset_path / "data").rglob("*.parquet"))

    if not parquet_files:
        raise ValueError(f"No parquet files found in {dataset_path / 'data'}")

    print(f"Computing statistics from {len(parquet_files)} files...")

    # Collect all data
    data_by_key: Dict[str, List] = {key: [] for key in keys}

    for i, pq_file in enumerate(parquet_files):
        if i % 100 == 0:
            print(f"  Processing file {i}/{len(parquet_files)}...")

        try:
            table = pq.read_table(pq_file)
            cols = table.column_names

            for key in keys:
                if key in cols:
                    col_data = table.column(key).to_pylist()
                    valid_data = [x for x in col_data if x is not None]
                    data_by_key[key].extend(valid_data)

        except Exception as e:
            print(f"  Warning: Error reading {pq_file}: {e}")

    # Compute statistics
    stats = {}

    for key, data_list in data_by_key.items():
        if not data_list:
            print(f"  No data for {key}")
            continue

        data = np.array(data_list, dtype=np.float32)
        print(f"{key}: shape {data.shape}")

        mean = np.mean(data, axis=0).tolist()
        std = np.std(data, axis=0).tolist()
        # Ensure std is not zero for normalization
        std = [max(s, 1e-6) for s in std]
        min_val = np.min(data, axis=0).tolist()
        max_val = np.max(data, axis=0).tolist()

        # Handle scalar case
        if isinstance(mean, float):
            mean, std, min_val, max_val = [mean], [std], [min_val], [max_val]

        stats[key] = {
            "mean": mean,
            "std": std,
            "min": min_val,
            "max": max_val,
        }

        print(f"  Computed stats for {len(mean)} dimensions")

    # Save if output file specified
    if output_file is None:
        output_file = dataset_path / "meta" / "stats.json"

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Saved statistics to {output_file}")

    return stats


def load_stats(dataset_path: Path) -> Dict[str, Dict[str, List[float]]]:
    """Load statistics from stats.json.

    Args:
        dataset_path: Path to dataset.

    Returns:
        Statistics dictionary.
    """
    stats_file = Path(dataset_path) / "meta" / "stats.json"

    if not stats_file.exists():
        raise FileNotFoundError(f"Stats file not found: {stats_file}")

    with open(stats_file) as f:
        return json.load(f)


def normalize(
    data: np.ndarray,
    stats: Dict[str, List[float]],
    method: str = "standard",
) -> np.ndarray:
    """Normalize data using computed statistics.

    Args:
        data: Data array to normalize.
        stats: Statistics dictionary with mean, std, min, max.
        method: Normalization method ("standard" or "minmax").

    Returns:
        Normalized data array.
    """
    if method == "standard":
        mean = np.array(stats["mean"])
        std = np.array(stats["std"])
        return (data - mean) / std

    elif method == "minmax":
        min_val = np.array(stats["min"])
        max_val = np.array(stats["max"])
        return (data - min_val) / (max_val - min_val + 1e-6)

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def denormalize(
    data: np.ndarray,
    stats: Dict[str, List[float]],
    method: str = "standard",
) -> np.ndarray:
    """Denormalize data using computed statistics.

    Args:
        data: Normalized data array.
        stats: Statistics dictionary.
        method: Normalization method used.

    Returns:
        Denormalized data array.
    """
    if method == "standard":
        mean = np.array(stats["mean"])
        std = np.array(stats["std"])
        return data * std + mean

    elif method == "minmax":
        min_val = np.array(stats["min"])
        max_val = np.array(stats["max"])
        return data * (max_val - min_val) + min_val

    else:
        raise ValueError(f"Unknown normalization method: {method}")
