"""
Download wildlife camera-trap images from LILA BC (NACTI) or HuggingFace (WCS).

Usage:
    python 01_download_dataset.py --species pronghorn --source nacti --out ../datasets/
    python 01_download_dataset.py --species pronghorn --source wcs   --out ../datasets/
    python 01_download_dataset.py --species all       --source nacti --out ../datasets/

NACTI requires gsutil (GCP) or the AWS CLI to be configured.
WCS is downloaded via HuggingFace datasets — no CLI tool required.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Species configuration
# ---------------------------------------------------------------------------

SPECIES_CONFIG: dict[str, dict] = {
    "pronghorn": {
        "nacti_category": "pronghorn",
        "wcs_query": "Antilocapra americana",
        "inaturalist_taxon_id": 42471,
    },
    "bighorn_sheep": {
        "nacti_category": "bighorn",
        "wcs_query": "Ovis canadensis",
        "inaturalist_taxon_id": 43218,
    },
    "golden_eagle": {
        "nacti_category": None,  # not in NACTI
        "wcs_query": "Aquila chrysaetos",
        "inaturalist_taxon_id": 5074,
    },
    "bison": {
        "nacti_category": "bison",
        "wcs_query": "Bison bison",
        "inaturalist_taxon_id": 43121,
    },
    "mule_deer": {
        "nacti_category": "mule deer",
        "wcs_query": "Odocoileus hemionus",
        "inaturalist_taxon_id": 42390,
    },
    "elk": {
        "nacti_category": "elk",
        "wcs_query": "Cervus canadensis",
        "inaturalist_taxon_id": 42418,
    },
    "coyote": {
        "nacti_category": "coyote",
        "wcs_query": "Canis latrans",
        "inaturalist_taxon_id": 41944,
    },
}

NACTI_GCP_BUCKET = "gs://public-datasets-lila/nacti/nacti_images/"
NACTI_MANIFEST_URL = "https://lila.science/wp-content/uploads/2023/09/nacti_metadata.json.zip"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_cmd(cmd: str) -> None:
    if shutil.which(cmd) is None:
        print(f"[error] '{cmd}' not found in PATH. Install it first.", file=sys.stderr)
        sys.exit(1)


def _mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# NACTI download
# ---------------------------------------------------------------------------

def download_nacti(species: list[str], out_dir: Path, max_images: int | None) -> None:
    """
    Download images from NACTI (North American Camera Trap Images) on LILA BC.

    Strategy:
      1. Download the NACTI metadata JSON (category list + annotations).
      2. Filter image IDs matching the requested species categories.
      3. Copy matching images from GCP or AWS using gsutil / aws s3 cp.

    Full dataset is ~400 GB — use --max-images to take a subset.
    """
    _require_cmd("gsutil")

    meta_dir = out_dir / "_nacti_meta"
    _mkdir(meta_dir)

    meta_path = meta_dir / "nacti_metadata.json"
    if not meta_path.exists():
        print("[nacti] Downloading metadata …")
        zip_path = meta_dir / "nacti_metadata.json.zip"
        subprocess.run(
            ["curl", "-L", "-o", str(zip_path), NACTI_MANIFEST_URL],
            check=True,
        )
        subprocess.run(["unzip", "-q", str(zip_path), "-d", str(meta_dir)], check=True)

    print("[nacti] Parsing metadata …")
    with open(meta_path) as f:
        meta = json.load(f)

    # Build category_id → species name map
    cat_map: dict[int, str] = {c["id"]: c["name"].lower() for c in meta["categories"]}

    # Target category IDs
    target_cats: set[int] = set()
    for sp in species:
        cfg = SPECIES_CONFIG.get(sp, {})
        nacti_cat = cfg.get("nacti_category")
        if nacti_cat is None:
            print(f"[nacti] '{sp}' not in NACTI — skipping.")
            continue
        for cat_id, cat_name in cat_map.items():
            if nacti_cat.lower() in cat_name:
                target_cats.add(cat_id)

    if not target_cats:
        print("[nacti] No matching categories found.")
        return

    # Map image_id → annotation category
    img_to_cat: dict[int, int] = {}
    for ann in meta["annotations"]:
        if ann["category_id"] in target_cats:
            img_to_cat[ann["image_id"]] = ann["category_id"]

    # Build list of (gcp_path, local_path)
    id_to_img: dict[int, dict] = {img["id"]: img for img in meta["images"]}
    tasks: list[tuple[str, Path]] = []
    for img_id, cat_id in img_to_cat.items():
        img = id_to_img.get(img_id)
        if img is None:
            continue
        sp_name = next(
            (sp for sp in species if SPECIES_CONFIG[sp].get("nacti_category", "").lower()
             in cat_map[cat_id]), species[0]
        )
        rel_path = img.get("file_name", img.get("url", "").split("/")[-1])
        local = out_dir / sp_name / "raw" / Path(rel_path).name
        gcp = NACTI_GCP_BUCKET + rel_path
        tasks.append((gcp, local))

    if max_images:
        tasks = tasks[:max_images]

    print(f"[nacti] Downloading {len(tasks)} images …")
    for gcp_path, local_path in tasks:
        if local_path.exists():
            continue
        _mkdir(local_path.parent)
        subprocess.run(["gsutil", "-q", "cp", gcp_path, str(local_path)], check=False)

    print("[nacti] Done.")


# ---------------------------------------------------------------------------
# WCS Camera Traps download (HuggingFace)
# ---------------------------------------------------------------------------

def download_wcs(species: list[str], out_dir: Path, max_images: int | None) -> None:
    """
    Download from WCS Camera Traps via HuggingFace datasets library.

    The dataset is filtered by the species scientific name present in the
    'species' column. Images are saved as JPEG under out_dir/<species>/raw/.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        print("[error] Install 'datasets': pip install datasets", file=sys.stderr)
        sys.exit(1)

    from PIL import Image as PILImage  # type: ignore

    for sp in species:
        query = SPECIES_CONFIG[sp]["wcs_query"]
        print(f"[wcs] Loading WCS Camera Traps for '{sp}' ({query}) …")

        ds = load_dataset(
            "society-ethics/lila_camera_traps",
            split="train",
            streaming=True,
        )
        ds = ds.filter(lambda row: query.lower() in (row.get("species") or "").lower())

        raw_dir = _mkdir(out_dir / sp / "raw")
        count = 0
        for row in ds:
            if max_images and count >= max_images:
                break
            img_obj = row.get("image")
            if img_obj is None:
                continue
            dest = raw_dir / f"{row['id']}.jpg"
            if not dest.exists():
                if isinstance(img_obj, PILImage.Image):
                    img_obj.save(dest)
                else:
                    dest.write_bytes(img_obj)
            count += 1

        print(f"[wcs] '{sp}': {count} images saved to {raw_dir}")


# ---------------------------------------------------------------------------
# iNaturalist GBIF export download
# ---------------------------------------------------------------------------

def download_inaturalist(species: list[str], out_dir: Path, max_images: int | None) -> None:
    """
    Download research-grade iNaturalist observations from GBIF.

    This uses the GBIF occurrence download API (free, no auth required for
    anonymous downloads up to 100K records). Images are fetched from the
    occurrence media URLs.

    Note: iNaturalist observations have species labels but NO bounding boxes.
    Run 02_prepare_dataset.py with --megadetector to generate bboxes.
    """
    import requests  # type: ignore

    GBIF_API = "https://api.gbif.org/v1/occurrence/search"

    for sp in species:
        taxon_id = SPECIES_CONFIG[sp]["inaturalist_taxon_id"]
        raw_dir = _mkdir(out_dir / sp / "raw")
        print(f"[inaturalist] Fetching '{sp}' (taxon {taxon_id}) …")

        offset, count = 0, 0
        limit = 100
        while True:
            if max_images and count >= max_images:
                break
            resp = requests.get(GBIF_API, params={
                "taxonKey": taxon_id,
                "datasetKey": "50c9509d-22c7-4a22-a47d-8c48425ef4a7",  # iNaturalist
                "hasCoordinate": "true",
                "qualityGrade": "research",
                "mediaType": "StillImage",
                "limit": limit,
                "offset": offset,
            }, timeout=30)
            data = resp.json()
            results = data.get("results", [])
            if not results:
                break

            for occ in results:
                if max_images and count >= max_images:
                    break
                for media in occ.get("media", []):
                    img_url = media.get("identifier")
                    if not img_url:
                        continue
                    key = occ["key"]
                    dest = raw_dir / f"{key}.jpg"
                    if dest.exists():
                        count += 1
                        break
                    try:
                        r = requests.get(img_url, timeout=15)
                        if r.status_code == 200:
                            dest.write_bytes(r.content)
                            count += 1
                    except Exception:
                        pass
                    break  # one image per occurrence

            offset += limit
            if data.get("endOfRecords"):
                break

        print(f"[inaturalist] '{sp}': {count} images saved to {raw_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download wildlife camera-trap datasets")
    p.add_argument(
        "--species",
        nargs="+",
        default=["pronghorn"],
        help="Species to download. Use 'all' for every configured species.",
    )
    p.add_argument(
        "--source",
        choices=["nacti", "wcs", "inaturalist", "all"],
        default="nacti",
        help="Data source (default: nacti)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("../datasets"),
        help="Root output directory (default: ../datasets)",
    )
    p.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Cap number of images per species (useful for quick tests)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    species_list: list[str] = (
        list(SPECIES_CONFIG.keys()) if "all" in args.species else args.species
    )
    unknown = [s for s in species_list if s not in SPECIES_CONFIG]
    if unknown:
        print(f"[error] Unknown species: {unknown}", file=sys.stderr)
        print(f"Available: {list(SPECIES_CONFIG.keys())}", file=sys.stderr)
        sys.exit(1)

    out_dir = args.out.resolve()
    _mkdir(out_dir)
    print(f"[info] Output directory: {out_dir}")
    print(f"[info] Species: {species_list}")
    print(f"[info] Source: {args.source}")

    if args.source in ("nacti", "all"):
        download_nacti(species_list, out_dir, args.max_images)
    if args.source in ("wcs", "all"):
        download_wcs(species_list, out_dir, args.max_images)
    if args.source in ("inaturalist", "all"):
        download_inaturalist(species_list, out_dir, args.max_images)


if __name__ == "__main__":
    main()
