"""
Auto-download all datasets for the experiment.
CWRU: auto-downloaded by preprocessing script
C-MAPSS: downloaded from NASA PCoE mirror
TEP: downloaded from GitHub mirror
"""

import os
import zipfile
import requests
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


def download_file(url: str, dest: Path, desc: str = ""):
    """Download a file with progress indicator."""
    print(f"  Downloading {desc or dest.name}...")
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    downloaded = 0

    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = downloaded / total * 100
                print(f"\r  {pct:.0f}% ({downloaded // 1024}KB / {total // 1024}KB)", end="", flush=True)
    print(f"\r  Done: {dest}")


def download_cmapss():
    """Download NASA C-MAPSS dataset."""
    raw_dir = DATA_DIR / "cmapss" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if (raw_dir / "train_FD001.txt").exists():
        print("[C-MAPSS] Already downloaded, skipping.")
        return

    print("[C-MAPSS] Downloading NASA Turbofan Engine Degradation dataset...")

    # Mirror from NASA PCoE - commonly used direct link
    zip_path = raw_dir / "CMAPSSData.zip"

    # Try multiple sources
    urls = [
        "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip",
        "https://ti.arc.nasa.gov/c/6/",
    ]

    downloaded = False
    for url in urls:
        try:
            download_file(url, zip_path, "C-MAPSS from NASA")
            downloaded = True
            break
        except Exception as e:
            print(f"  Source failed: {e}, trying next...")

    if not downloaded:
        print("  Auto-download failed. Please download manually:")
        print("  1. Go to: https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6")
        print("  2. Click 'Export' -> Download")
        print(f"  3. Extract to: {raw_dir}/")
        return

    # Extract
    print("  Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            filename = os.path.basename(member)
            if filename and filename.endswith(".txt"):
                with zf.open(member) as src, open(raw_dir / filename, "wb") as dst:
                    dst.write(src.read())

    zip_path.unlink()
    print(f"  Extracted to {raw_dir}")

    # Verify
    if (raw_dir / "train_FD001.txt").exists():
        print("  [OK] train_FD001.txt found")
    else:
        # List what we got
        files = list(raw_dir.glob("*"))
        print(f"  Files found: {[f.name for f in files]}")
        print("  Note: file names may differ. Check and rename if needed.")


def download_tep():
    """Download Tennessee Eastman Process dataset."""
    raw_dir = DATA_DIR / "tep" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if (raw_dir / "d00.dat").exists():
        print("[TEP] Already downloaded, skipping.")
        return

    print("[TEP] Downloading Tennessee Eastman Process dataset...")

    # GitHub mirror (Rieth et al. version, most commonly used)
    base_url = "https://raw.githubusercontent.com/camaramm/tennessee-eastman-profBraworkers/master/datasets/TEP_FaultFree_Training.RData"

    # Try the simpler .dat format from another mirror
    dat_urls = {
        "d00.dat": "https://raw.githubusercontent.com/makinarocks/awesome-industrial-machine-datasets/master/data-list/Tennessee-Eastman-Process/data/d00.dat",
    }

    # Try downloading individual fault files from Harvard Dataverse API
    print("  Attempting download from Harvard Dataverse...")
    try:
        # Harvard Dataverse API for the TEP dataset
        api_url = "https://dataverse.harvard.edu/api/access/dataset/:persistentId?persistentId=doi:10.7910/DVN/6C3JR1"
        zip_path = raw_dir / "tep_data.zip"
        download_file(api_url, zip_path, "TEP from Harvard Dataverse")

        print("  Extracting...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.namelist():
                filename = os.path.basename(member)
                if filename and (filename.endswith(".dat") or filename.endswith(".csv")):
                    with zf.open(member) as src, open(raw_dir / filename, "wb") as dst:
                        dst.write(src.read())
        zip_path.unlink()
        print(f"  Extracted to {raw_dir}")

    except Exception as e:
        print(f"  Harvard Dataverse download failed: {e}")
        print("")
        print("  Please download manually:")
        print("  1. Go to: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/6C3JR1")
        print("  2. Click 'Access Dataset' -> Download ZIP")
        print(f"  3. Extract .dat files to: {raw_dir}/")
        return

    # Verify
    files = list(raw_dir.glob("d*.dat")) + list(raw_dir.glob("d*.csv"))
    if files:
        print(f"  [OK] Found {len(files)} data files")
    else:
        files_all = list(raw_dir.glob("*"))
        print(f"  Files found: {[f.name for f in files_all[:20]]}")
        print("  Note: may need to rename files to d00.dat, d01.dat, etc.")


def main():
    print("=" * 50)
    print("Downloading datasets for LLM-PDM experiment")
    print("=" * 50)
    print()

    print("[CWRU] Will auto-download during preprocessing (no action needed)")
    print()

    download_cmapss()
    print()

    download_tep()
    print()

    print("=" * 50)
    print("Done! Next step: run preprocessing")
    print("  python -m src.preprocessing.cwru_preprocess")
    print("  python -m src.preprocessing.cmapss_preprocess")
    print("  python -m src.preprocessing.tep_preprocess")
    print("=" * 50)


if __name__ == "__main__":
    main()
