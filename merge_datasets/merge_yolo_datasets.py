#!/usr/bin/env python3
"""
USAGE:

python merge_yolo_datasets.py \
  --source_root /path/to/all_datasets_root \
  --output_root /path/to/merged_dataset \
  --train 0.8 --val 0.1 --test 0.1 \
  --mode copy \
  --prefix dataset
  --ignore_names "person,traffic light,license_plate"

"""
import argparse
import os
import sys
import shutil
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Iterable, Set

# Optional YAML support
try:
    import yaml  # type: ignore
except Exception:
    yaml = None

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def iter_files(path: Path):
    if not path.exists():
        return []
    return (p for p in path.rglob("*") if p.is_file())


def is_image_candidate(p: Path) -> bool:
    # Accept known image extensions OR extensionless files
    suf = p.suffix.lower()
    return (suf in IMAGE_EXTS) or (suf == "")


def find_images_in_split(ds: Path, split: str) -> List[Path]:
    images_dir = ds / split / "images"
    if not images_dir.exists():
        return []
    return [p for p in iter_files(images_dir) if is_image_candidate(p)]


def label_path_for_image(img_path: Path) -> Path:
    parts = list(img_path.parts)
    try:
        idx = parts.index("images")
    except ValueError:
        return img_path.with_suffix(".txt")
    labels_parts = parts.copy()
    labels_parts[idx] = "labels"
    lbl = Path(*labels_parts)
    if lbl.suffix:
        lbl = lbl.with_suffix(".txt")
    else:
        lbl = lbl.with_name(lbl.name + ".txt")
    return lbl


def load_dataset_names(ds: Path) -> Optional[Dict[int, str]]:
    """Load class names mapping for a dataset (index -> name) from its data.yaml"""
    if yaml is None:
        return None
    dy = ds / "data.yaml"
    if not dy.exists():
        return None
    try:
        data = yaml.safe_load(dy.read_text(encoding="utf-8"))
        names = data.get("names")
        if isinstance(names, dict):
            out = {}
            for k, v in names.items():
                try:
                    out[int(k)] = str(v)
                except Exception:
                    return None
            return out
        elif isinstance(names, list):
            return {i: str(n) for i, n in enumerate(names)}
    except Exception:
        return None
    return None


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def safe_transfer(src: Path, dst: Path, mode: str = "copy"):
    ensure_dir(dst.parent)
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        try:
            if dst.exists():
                dst.unlink()
            os.link(src, dst)
        except Exception:
            shutil.copy2(src, dst)
    elif mode == "symlink":
        try:
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            os.symlink(src, dst)
        except Exception:
            shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def unique_name(prefix: Optional[str], original_name: str, avoid: set) -> str:
    base = original_name
    if prefix:
        base = f"{prefix}_{base}"
    name = base
    stem = Path(base).stem
    suffix = Path(base).suffix
    i = 1
    while name in avoid:
        name = f"{stem}_{i}{suffix}"
        i += 1
    avoid.add(name)
    return name


def global_split(items, train: float, val: float, test: float, seed: int):
    assert abs((train + val + test) - 1.0) < 1e-6, "Splits must sum to 1.0"
    rnd = random.Random(seed)
    rnd.shuffle(items)
    n = len(items)
    n_train = int(n * train)
    n_val = int(n * val)
    return {
        "train": items[:n_train],
        "val": items[n_train:n_train + n_val],
        "test": items[n_train + n_val:]
    }


def collect_items(source_root: Path, prefix_mode: str = "dataset"):
    """
    Gather items and per-dataset stats, plus per-dataset class names.
    Returns:
      items: list of (img_path, lbl_path, dataset_prefix, split_hint, dataset_root)
      stats: dict name -> counters
      ds_names: dict dataset_root -> {class_id: class_name}
    """
    items = []
    stats = {}
    ds_names = {}
    for ds in sorted([p for p in source_root.iterdir() if p.is_dir()]):
        dataset_name = ds.name
        dataset_prefix = ds.name if prefix_mode == "dataset" else ""
        stats[dataset_name] = {"total": 0, "train": 0, "valid": 0, "val": 0, "test": 0}
        names_map = load_dataset_names(ds)  # may be None
        ds_names[ds] = names_map

        for split in ["train", "valid", "val", "test"]:
            imgs = find_images_in_split(ds, split)
            count_split_valid = 0
            for img in imgs:
                lbl = label_path_for_image(img)
                if lbl.exists():
                    items.append((img, lbl, dataset_prefix, split, ds))
                    count_split_valid += 1
            stats[dataset_name][split] += count_split_valid
            stats[dataset_name]["total"] += count_split_valid

    return items, stats, ds_names


def write_data_yaml(out_root: Path):
    """Output unified data.yaml with single class 'car'"""
    content = {
        "path": str(out_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": ["car"],
    }
    if yaml is None:
        text = "path: {}\ntrain: {}\nval: {}\ntest: {}\nnames: {}\n".format(
            content["path"], content["train"], content["val"], content["test"], content["names"]
        )
        (out_root / "data.yaml").write_text(text, encoding="utf-8")
    else:
        with open(out_root / "data.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(content, f, sort_keys=False, allow_unicode=True)


def print_dataset_stats(stats):
    print("\nPer-dataset stats (only items with existing labels are counted):")
    for ds_name in sorted(stats.keys()):
        s = stats[ds_name]
        print(f" - {ds_name}: total={s['total']}, train={s['train']}, valid={s['valid']}, val={s['val']}, test={s['test']}")


def print_class_names(ds_names: Dict[Path, Optional[Dict[int, str]]]):
    print("\nPer-dataset class names (from data.yaml):")
    global_names: Set[str] = set()
    for ds, mapping in ds_names.items():
        if mapping is None:
            print(f" - {ds.name}: <no data.yaml or unable to read>")
            continue
        ordered = [mapping[k] for k in sorted(mapping.keys())]
        print(f" - {ds.name}: {ordered}")
        for n in ordered:
            global_names.add(n)
    print("\nGlobal union of class names:")
    if global_names:
        print(" ", sorted(global_names))
    else:
        print("  <empty>")


def remap_label_lines(lines: List[str], names_map: Optional[Dict[int, str]], ignore: Set[str]) -> List[str]:
    """
    Map all non-ignored classes to class 0 ('car').
    If a class name is in ignore set, drop that line.
    If names_map is None or id not in map, treat it as non-ignored and map to 0.
    """
    out = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split()
        try:
            cid = int(parts[0])
        except Exception:
            # malformed line -> skip
            continue
        cname = None
        if names_map is not None:
            cname = names_map.get(cid)
        # If class name is known and in ignore, drop
        if cname is not None and cname in ignore:
            continue
        # Otherwise map to 0
        parts[0] = "0"
        out.append(" ".join(parts))
    return out


def main():
    ap = argparse.ArgumentParser(description="Merge YOLO datasets into one, remapping classes to single 'car' unless ignored.")
    ap.add_argument("--source_root", required=True, type=Path,
                    help="Folder containing multiple YOLO datasets as subfolders.")
    ap.add_argument("--output_root", required=True, type=Path,
                    help="Where to create the merged dataset.")
    ap.add_argument("--train", type=float, default=0.8, help="Train ratio (0..1)")
    ap.add_argument("--val", type=float, default=0.1, help="Val ratio (0..1)")
    ap.add_argument("--test", type=float, default=0.1, help="Test ratio (0..1)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    ap.add_argument("--mode", choices=["copy", "hardlink", "symlink"], default="copy",
                    help="How to place files into the merged dataset.")
    ap.add_argument("--prefix", choices=["dataset", "none"], default="dataset",
                    help="Prefix output filenames with dataset folder name to avoid collisions.")
    ap.add_argument("--ignore_names", type=str, default=None,
                    help="Comma-separated class names to IGNORE (not mapped to 'car'). Others map to 'car'.")
    args = ap.parse_args()

    total = args.train + args.val + args.test
    if abs(total - 1.0) > 1e-6:
        print(f"Error: splits must sum to 1.0 (got {total})", file=sys.stderr)
        sys.exit(1)

    source_root = args.source_root
    out_root = args.output_root

    if not source_root.exists():
        print(f"Error: source_root not found: {source_root}", file=sys.stderr)
        sys.exit(1)

    if out_root.exists():
        print(f"Removing existing output folder: {out_root}")
        shutil.rmtree(out_root)

    # Parse ignore set
    ignore_set: Set[str] = set()
    if args.ignore_names:
        ignore_set = {n.strip() for n in args.ignore_names.split(",") if n.strip()}
    print(f"Ignore class names: {sorted(ignore_set) if ignore_set else '[]'}")

    # Collect items and stats and per-dataset names map
    print("Scanning datasets...")
    items, per_ds_stats, ds_names = collect_items(source_root, prefix_mode=args.prefix)

    if not items:
        print("No labeled images found under any dataset's '<split>/images' folders.", file=sys.stderr)
        print_dataset_stats(per_ds_stats)
        sys.exit(1)

    # Print class names for debugging
    print_class_names(ds_names)

    # Report per-dataset stats
    print_dataset_stats(per_ds_stats)

    # Global split
    print("\nShuffling and splitting globally...")
    splits = global_split(items, args.train, args.val, args.test, seed=args.seed)

    # Prepare output dirs
    img_train = out_root / "images" / "train"
    img_val = out_root / "images" / "val"
    img_test = out_root / "images" / "test"
    lbl_train = out_root / "labels" / "train"
    lbl_val = out_root / "labels" / "val"
    lbl_test = out_root / "labels" / "test"
    for d in [img_train, img_val, img_test, lbl_train, lbl_val, lbl_test]:
        d.mkdir(parents=True, exist_ok=True)

    # Track used basenames to avoid collisions
    used_names = {"train": set(), "val": set(), "test": set()}

    def place_items(items_list, split: str):
        count = 0
        kept = 0
        for img, lbl, ds_prefix, _, ds_root in items_list:
            prefix = ds_prefix if args.prefix == "dataset" and ds_prefix else None
            img_name = unique_name(prefix, Path(img).name, used_names[split])
            lbl_name = Path(img_name).with_suffix(".txt").name

            # Read, remap, and maybe drop the label if empty after ignore
            try:
                lines = lbl.read_text(encoding="utf-8").splitlines()
            except Exception:
                continue
            mapping = ds_names.get(ds_root)  # may be None
            remapped = remap_label_lines(lines, mapping, ignore_set)
            if not remapped:
                # No boxes left after filtering -> drop this image from merged set
                continue

            # Transfer image
            if split == "train":
                img_dst = img_train / img_name
                lbl_dst = lbl_train / lbl_name
            elif split == "val":
                img_dst = img_val / img_name
                lbl_dst = lbl_val / lbl_name
            else:
                img_dst = img_test / img_name
                lbl_dst = lbl_test / lbl_name

            safe_transfer(img, img_dst, mode=args.mode)

            # Write remapped label
            ensure_dir(lbl_dst.parent)
            with open(lbl_dst, "w", encoding="utf-8") as f:
                f.write("\n".join(remapped) + "\n")

            count += 1
            kept += 1
        return kept

    print("Placing train files...")
    n_tr = place_items(splits["train"], "train")
    print("Placing val files...")
    n_vl = place_items(splits["val"], "val")
    print("Placing test files...")
    n_ts = place_items(splits["test"], "test")

    # Write data.yaml with single class 'car'
    write_data_yaml(out_root)

    print("\nMerged dataset stats (after remap to 'car' and ignoring selected classes):")
    print(f" total={n_tr + n_vl + n_ts}, train={n_tr}, val={n_vl}, test={n_ts}")
    print("\nDone.")
    print(f"Merged dataset created at: {out_root}")
    print("Output data.yaml uses a single class: ['car']")

if __name__ == "__main__":
    main()
