#!/usr/bin/env python3
"""
Randomly move N files from a primary directory and move corresponding files
from other directories into their paired destination directories.

Example:
  python tools/random_move_images.py \
    --count 50 \
    --pair data/images/train data/images/val \
    --pair data/masks/train data/masks/val \
    --primary-index 0 \
    --pattern "*.jpg" \
    --seed 123

Config file example (tools/random_move_images_config.json):
  {
    "count": 50,
    "pairs": [
      ["data/images/train", "data/images/val"],
      ["data/masks/train", "data/masks/val"]
    ],
    "primary_index": 0,
    "pattern": "*.jpg",
    "by_stem": true,
    "exts": ".png",
    "seed": 123,
    "allow_smaller": true
  }
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

DEFAULT_CONFIG_PATH = Path(__file__).with_name("random_move_images_config.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Randomly move N files from a primary directory and move corresponding "
            "files from other directories into their paired destinations."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Path to a JSON config file. If omitted and "
            "tools/random_move_images_config.json exists, it is used."
        ),
    )
    parser.add_argument(
        "--count",
        "-n",
        type=int,
        default=None,
        help="Number of files to move from the primary directory.",
    )
    parser.add_argument(
        "--pair",
        nargs=2,
        action="append",
        metavar=("SRC", "DST"),
        default=None,
        help="Source and destination directory pair. Can be repeated.",
    )
    parser.add_argument(
        "--primary-index",
        type=int,
        default=None,
        help="Index of the --pair entry to use for random selection (0-based).",
    )
    parser.add_argument(
        "--pattern",
        default=None,
        help="Glob pattern for selecting files in the primary source directory.",
    )
    parser.add_argument(
        "--by-stem",
        action="store_true",
        default=None,
        help="Match corresponding files in other dirs by filename stem.",
    )
    parser.add_argument(
        "--exts",
        default=None,
        help=(
            "Comma-separated extensions to use with --by-stem (e.g. .png,.jpg). "
            "If empty, any extension is allowed."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--allow-smaller",
        action="store_true",
        default=None,
        help="If N is larger than available files, move all available files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=None,
        help="Overwrite existing files in destination directories.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=None,
        help="Skip files when the destination file already exists.",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        default=None,
        help="Skip files when a corresponding source file is missing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=None,
        help="Print planned moves without modifying files.",
    )
    return parser.parse_args()


def config_get(config: dict, *keys: str):
    for key in keys:
        if key in config:
            return config[key]
    return None


def coerce_int(value, name: str) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Config '{name}' must be an integer.") from exc


def coerce_bool(value, name: str) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    raise ValueError(f"Config '{name}' must be a boolean.")


def coerce_str(value, name: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    raise ValueError(f"Config '{name}' must be a string.")


def coerce_exts(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    if isinstance(value, str):
        return value
    raise ValueError("Config 'exts' must be a string or list.")


def normalize_pairs(pairs_value) -> list[list[str]] | None:
    if pairs_value is None:
        return None
    if not isinstance(pairs_value, list):
        raise ValueError("Config 'pairs' must be a list of [src, dst] entries.")
    normalized: list[list[str]] = []
    for idx, item in enumerate(pairs_value):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"Config 'pairs' entry {idx} must be [src, dst].")
        src, dst = item
        if not isinstance(src, str) or not isinstance(dst, str):
            raise ValueError(f"Config 'pairs' entry {idx} must contain strings.")
        normalized.append([src, dst])
    return normalized


def load_config(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"Config file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in config file {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a JSON object: {path}")
    return data


def apply_config(args: argparse.Namespace, config: dict) -> argparse.Namespace:
    config_pairs = normalize_pairs(config_get(config, "pairs", "pair"))
    if args.pair is None:
        args.pair = config_pairs
    args.count = coerce_int(
        args.count if args.count is not None else config_get(config, "count"),
        "count",
    )
    args.primary_index = coerce_int(
        args.primary_index
        if args.primary_index is not None
        else config_get(config, "primary_index", "primary-index"),
        "primary_index",
    )
    args.pattern = coerce_str(
        args.pattern if args.pattern is not None else config_get(config, "pattern"),
        "pattern",
    )
    args.by_stem = coerce_bool(
        args.by_stem if args.by_stem is not None else config_get(config, "by_stem", "by-stem"),
        "by_stem",
    )
    args.exts = coerce_exts(
        args.exts if args.exts is not None else config_get(config, "exts")
    )
    args.seed = coerce_int(
        args.seed if args.seed is not None else config_get(config, "seed"),
        "seed",
    )
    args.allow_smaller = coerce_bool(
        args.allow_smaller
        if args.allow_smaller is not None
        else config_get(config, "allow_smaller", "allow-smaller"),
        "allow_smaller",
    )
    args.overwrite = coerce_bool(
        args.overwrite if args.overwrite is not None else config_get(config, "overwrite"),
        "overwrite",
    )
    args.skip_existing = coerce_bool(
        args.skip_existing
        if args.skip_existing is not None
        else config_get(config, "skip_existing", "skip-existing"),
        "skip_existing",
    )
    args.skip_missing = coerce_bool(
        args.skip_missing
        if args.skip_missing is not None
        else config_get(config, "skip_missing", "skip-missing"),
        "skip_missing",
    )
    args.dry_run = coerce_bool(
        args.dry_run if args.dry_run is not None else config_get(config, "dry_run", "dry-run"),
        "dry_run",
    )
    return args


def parse_exts(exts: str) -> list[str]:
    if not exts:
        return []
    parts = [p.strip() for p in exts.split(",") if p.strip()]
    normalized = []
    for part in parts:
        normalized.append(part if part.startswith(".") else f".{part}")
    return normalized


def select_primary_files(
    primary_dir: Path, pattern: str, count: int, seed: int | None, allow_smaller: bool
) -> list[Path]:
    files = sorted(p for p in primary_dir.glob(pattern) if p.is_file())
    if not files:
        raise ValueError(f"No files found in {primary_dir} with pattern {pattern!r}.")
    if count <= 0:
        raise ValueError("--count must be > 0.")
    if count > len(files):
        if allow_smaller:
            count = len(files)
        else:
            raise ValueError(
                f"Requested {count} files but only {len(files)} available in {primary_dir}."
            )
    rng = random.Random(seed)
    return rng.sample(files, count)


def resolve_corresponding_file(
    src_dir: Path,
    primary_file: Path,
    by_stem: bool,
    exts: list[str],
) -> Path | None:
    if not by_stem:
        candidate = src_dir / primary_file.name
        return candidate if candidate.is_file() else None

    stem = primary_file.stem
    if exts:
        candidates = [src_dir / f"{stem}{ext}" for ext in exts]
        candidates = [p for p in candidates if p.is_file()]
    else:
        candidates = [p for p in src_dir.glob(f"{stem}.*") if p.is_file()]

    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValueError(
            f"Multiple matches for stem {stem!r} in {src_dir}: "
            + ", ".join(p.name for p in candidates)
        )
    return None


def main() -> int:
    args = parse_args()
    config: dict = {}
    config_path: Path | None = None

    if args.config:
        config_path = Path(args.config).expanduser()
    elif DEFAULT_CONFIG_PATH.is_file():
        config_path = DEFAULT_CONFIG_PATH

    if config_path:
        try:
            config = load_config(config_path)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2

    try:
        args = apply_config(args, config)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.count is None:
        print("--count is required (or set 'count' in config).", file=sys.stderr)
        return 2
    if args.pair is None:
        print("--pair is required (or set 'pairs' in config).", file=sys.stderr)
        return 2

    if args.primary_index is None:
        args.primary_index = 0
    if args.pattern is None:
        args.pattern = "*"
    if args.by_stem is None:
        args.by_stem = False
    if args.exts is None:
        args.exts = ""
    if args.allow_smaller is None:
        args.allow_smaller = False
    if args.overwrite is None:
        args.overwrite = False
    if args.skip_existing is None:
        args.skip_existing = False
    if args.skip_missing is None:
        args.skip_missing = False
    if args.dry_run is None:
        args.dry_run = False

    pairs = [(Path(src).expanduser(), Path(dst).expanduser()) for src, dst in args.pair]
    if args.primary_index < 0 or args.primary_index >= len(pairs):
        print("--primary-index is out of range for --pair entries.", file=sys.stderr)
        return 2

    for src_dir, dst_dir in pairs:
        if not src_dir.is_dir():
            print(f"Source directory does not exist: {src_dir}", file=sys.stderr)
            return 2
        if src_dir.resolve() == dst_dir.resolve():
            print(f"Source and destination are the same: {src_dir}", file=sys.stderr)
            return 2

    primary_dir = pairs[args.primary_index][0]
    exts = parse_exts(args.exts)

    try:
        selected = select_primary_files(
            primary_dir, args.pattern, args.count, args.seed, args.allow_smaller
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    planned_moves: list[tuple[Path, Path, bool]] = []
    errors: list[str] = []
    skipped: list[str] = []

    for primary_file in selected:
        for idx, (src_dir, dst_dir) in enumerate(pairs):
            if idx == args.primary_index:
                src_file = primary_file
            else:
                try:
                    src_file = resolve_corresponding_file(
                        src_dir, primary_file, args.by_stem, exts
                    )
                except ValueError as exc:
                    errors.append(str(exc))
                    continue

            if src_file is None or not src_file.is_file():
                msg = f"Missing source file for {primary_file.name} in {src_dir}"
                if args.skip_missing:
                    skipped.append(msg)
                else:
                    errors.append(msg)
                continue

            dest_file = dst_dir / src_file.name
            if src_file.resolve() == dest_file.resolve():
                errors.append(f"Source and destination are the same file: {src_file}")
                continue

            if dest_file.exists():
                if args.overwrite:
                    planned_moves.append((src_file, dest_file, True))
                elif args.skip_existing:
                    skipped.append(f"Destination exists, skipping: {dest_file}")
                else:
                    errors.append(f"Destination already exists: {dest_file}")
            else:
                planned_moves.append((src_file, dest_file, False))

    if errors:
        print("Errors encountered:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 2

    if args.dry_run:
        for src_file, dest_file, overwrite in planned_moves:
            flag = "overwrite" if overwrite else "move"
            print(f"[dry-run] {flag}: {src_file} -> {dest_file}")
        if skipped:
            print("Skipped:", file=sys.stderr)
            for msg in skipped:
                print(f"  - {msg}", file=sys.stderr)
        return 0

    for _, dst_dir in pairs:
        dst_dir.mkdir(parents=True, exist_ok=True)

    for src_file, dest_file, overwrite in planned_moves:
        if overwrite and dest_file.exists():
            if dest_file.is_dir():
                print(f"Destination is a directory, skipping: {dest_file}", file=sys.stderr)
                continue
            dest_file.unlink()
        shutil.move(str(src_file), str(dest_file))

    if skipped:
        print("Skipped:", file=sys.stderr)
        for msg in skipped:
            print(f"  - {msg}", file=sys.stderr)

    print(f"Moved {len(planned_moves)} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
