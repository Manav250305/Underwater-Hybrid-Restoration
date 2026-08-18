"""
Module — UIEB Dataset Loader
Resolves paths to the UIEB raw-890 / reference-890 / challenging-60 image sets
and provides paired iteration for batch evaluation (see README section 8,
"Next Steps" #2: batch-evaluation script over data/raw/).

The full dataset (~2GB) lives on external storage rather than in the repo or
data/, so each split is resolved by checking, in order:

  1. Environment variable override
     (UIEB_RAW_DIR / UIEB_REFERENCE_DIR / UIEB_CHALLENGING_DIR)
  2. Local project dirs (data/raw, data/reference, data/challenging) —
     used if you copy/symlink images there instead
  3. External SSD mount (default paths below)

The first candidate that actually contains images wins, so this keeps
working whether the dataset sits on the SSD, has been copied locally, or
is somewhere else entirely (via the env vars) on another machine.
"""

import glob
import os

import cv2

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_DEFAULT_CANDIDATES = {
    "raw": [
        os.environ.get("UIEB_RAW_DIR"),
        os.path.join(_PROJECT_ROOT, "data", "raw"),
        "/Volumes/Lexar/raw-890",
    ],
    "reference": [
        os.environ.get("UIEB_REFERENCE_DIR"),
        os.path.join(_PROJECT_ROOT, "data", "reference"),
        "/Volumes/Lexar/reference-890",
    ],
    "challenging": [
        os.environ.get("UIEB_CHALLENGING_DIR"),
        os.path.join(_PROJECT_ROOT, "data", "challenging"),
        "/Volumes/Lexar/challenging-60",
    ],
}

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def _is_image(name: str) -> bool:
    # UIEB was copied from a FAT/exFAT-formatted drive, which leaves macOS
    # AppleDouble sidecar files (._foo.png) alongside the real ones — skip them.
    return not name.startswith(".") and name.lower().endswith(_IMAGE_EXTS)


def _list_images(dir_path: str) -> dict:
    """filename -> full path, for real image files only (skips ._ sidecars)."""
    return {
        os.path.basename(p): p
        for p in sorted(glob.glob(os.path.join(dir_path, "*")))
        if _is_image(os.path.basename(p))
    }


def resolve_split_dir(split: str) -> str:
    """
    Return the first existing, non-empty directory for a dataset split
    ('raw', 'reference', or 'challenging'): env override -> local data/ ->
    external SSD mount, in that order.
    """
    if split not in _DEFAULT_CANDIDATES:
        raise ValueError(f"Unknown split '{split}', expected one of {list(_DEFAULT_CANDIDATES)}")

    candidates = [c for c in _DEFAULT_CANDIDATES[split] if c]
    for c in candidates:
        if os.path.isdir(c) and _list_images(c):
            return c

    raise FileNotFoundError(
        f"Could not find UIEB '{split}' images in any of: {candidates}. "
        f"Mount the SSD with the UIEB dataset, place images in data/{split}/, "
        f"or set the UIEB_{split.upper()}_DIR env var. See README section 5."
    )


def list_pairs():
    """
    Return a sorted list of (filename, raw_path, reference_path) for every
    UIEB image present in both the raw and reference splits.
    """
    raw_files = _list_images(resolve_split_dir("raw"))
    ref_files = _list_images(resolve_split_dir("reference"))

    common = sorted(set(raw_files) & set(ref_files))
    n_missing_ref = len(raw_files) - len(common)
    n_missing_raw = len(ref_files) - len(common)
    if n_missing_ref or n_missing_raw:
        print(f"[dataset_loader] warning: {n_missing_ref} raw images have no reference match, "
              f"{n_missing_raw} reference images have no raw match — using {len(common)} paired images.")

    return [(name, raw_files[name], ref_files[name]) for name in common]


def list_challenging():
    """Return a sorted list of (filename, path) for the no-reference challenging-60 set."""
    return sorted(_list_images(resolve_split_dir("challenging")).items())


def load_pair(filename: str):
    """Load one (raw, reference) BGR image pair by filename, raising if either is missing/unreadable."""
    raw_path = os.path.join(resolve_split_dir("raw"), filename)
    ref_path = os.path.join(resolve_split_dir("reference"), filename)

    raw_img = cv2.imread(raw_path)
    ref_img = cv2.imread(ref_path)
    if raw_img is None:
        raise FileNotFoundError(f"Could not read raw image: {raw_path}")
    if ref_img is None:
        raise FileNotFoundError(f"Could not read reference image: {ref_path}")
    return raw_img, ref_img


def iter_pairs():
    """Yield (filename, raw_bgr, reference_bgr) for every paired UIEB image, loaded lazily."""
    for name, raw_path, ref_path in list_pairs():
        raw_img = cv2.imread(raw_path)
        ref_img = cv2.imread(ref_path)
        if raw_img is None or ref_img is None:
            print(f"[dataset_loader] warning: skipping unreadable pair '{name}'")
            continue
        yield name, raw_img, ref_img


if __name__ == "__main__":
    for split in ("raw", "reference", "challenging"):
        try:
            d = resolve_split_dir(split)
            print(f"{split:12s} -> {d}  ({len(_list_images(d))} images)")
        except FileNotFoundError as e:
            print(f"{split:12s} -> NOT FOUND ({e})")

    try:
        pairs = list_pairs()
        print(f"\n{len(pairs)} raw/reference pairs available for batch evaluation.")
        print(f"first pair: {pairs[0]}")
    except FileNotFoundError as e:
        print(e)
