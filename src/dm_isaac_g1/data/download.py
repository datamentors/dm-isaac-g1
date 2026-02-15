"""Dataset download utilities for HuggingFace datasets."""

import subprocess
from pathlib import Path
from typing import List, Optional

# Available hospitality datasets
HOSPITALITY_DATASETS = [
    "unitreerobotics/G1_Fold_Towel",
    "unitreerobotics/G1_Clean_Table",
    "unitreerobotics/G1_Wipe_Table",
    "unitreerobotics/G1_Prepare_Fruit",
    "unitreerobotics/G1_Pour_Medicine",
    "unitreerobotics/G1_Organize_Tools",
    "unitreerobotics/G1_Pack_PingPong",
]

# Dex3 datasets
DEX3_DATASETS = [
    "unitreerobotics/G1_Dex3_ToastedBread_Dataset",
    "unitreerobotics/G1_Dex3_BlockStacking_Dataset",
]


def download_dataset(
    repo_id: str,
    output_dir: Path,
    use_lfs: bool = True,
    hf_token: Optional[str] = None,
) -> Path:
    """Download a dataset from HuggingFace.

    Args:
        repo_id: HuggingFace repository ID (e.g., "unitreerobotics/G1_Fold_Towel").
        output_dir: Directory to download to.
        use_lfs: Use git LFS for large files (recommended for video datasets).
        hf_token: HuggingFace token for private repos.

    Returns:
        Path to downloaded dataset.

    Raises:
        RuntimeError: If download fails.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract dataset name from repo_id
    dataset_name = repo_id.split("/")[-1]
    dataset_path = output_dir / dataset_name

    if dataset_path.exists():
        print(f"Dataset already exists: {dataset_path}")
        return dataset_path

    if use_lfs:
        # Use git clone with LFS (better for large files)
        print(f"Cloning {repo_id} with git LFS...")

        # Ensure LFS is installed
        subprocess.run(["git", "lfs", "install"], check=True, capture_output=True)

        # Clone the repo
        clone_url = f"https://huggingface.co/datasets/{repo_id}"
        if hf_token:
            clone_url = f"https://user:{hf_token}@huggingface.co/datasets/{repo_id}"

        result = subprocess.run(
            ["git", "clone", clone_url, str(dataset_path)],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Git clone failed: {result.stderr}")

    else:
        # Use huggingface-cli
        print(f"Downloading {repo_id} with huggingface-cli...")

        cmd = [
            "huggingface-cli",
            "download",
            repo_id,
            "--repo-type",
            "dataset",
            "--local-dir",
            str(dataset_path),
        ]

        if hf_token:
            cmd.extend(["--token", hf_token])

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Download failed: {result.stderr}")

    print(f"Downloaded to: {dataset_path}")
    return dataset_path


def download_hospitality_datasets(
    output_dir: Path,
    datasets: Optional[List[str]] = None,
    use_lfs: bool = True,
    hf_token: Optional[str] = None,
) -> List[Path]:
    """Download all hospitality datasets.

    Args:
        output_dir: Directory to download to.
        datasets: Specific datasets to download. Downloads all if None.
        use_lfs: Use git LFS.
        hf_token: HuggingFace token.

    Returns:
        List of paths to downloaded datasets.
    """
    datasets = datasets or HOSPITALITY_DATASETS
    downloaded = []

    for repo_id in datasets:
        try:
            path = download_dataset(
                repo_id=repo_id,
                output_dir=output_dir,
                use_lfs=use_lfs,
                hf_token=hf_token,
            )
            downloaded.append(path)
        except Exception as e:
            print(f"Failed to download {repo_id}: {e}")

    return downloaded


def download_dex3_datasets(
    output_dir: Path,
    use_lfs: bool = True,
    hf_token: Optional[str] = None,
) -> List[Path]:
    """Download Dex3 datasets.

    Args:
        output_dir: Directory to download to.
        use_lfs: Use git LFS.
        hf_token: HuggingFace token.

    Returns:
        List of paths to downloaded datasets.
    """
    downloaded = []

    for repo_id in DEX3_DATASETS:
        try:
            path = download_dataset(
                repo_id=repo_id,
                output_dir=output_dir,
                use_lfs=use_lfs,
                hf_token=hf_token,
            )
            downloaded.append(path)
        except Exception as e:
            print(f"Failed to download {repo_id}: {e}")

    return downloaded
