#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys
from typing import List, Optional, Set, Tuple

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


LABELS = {
    "1": "not relevant",
    "2": "streaks",
    "3": "actual fp",
    "4": "hard to categorize",
}

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# Hard-coded configuration (edit these values as needed).
HARD_CONFIG = {
    # Example: "/path/to/images". Leave as None to provide via CLI.
    "input_dir": '/home/ckchng/Documents/realtime-semantic-segmentation-pytorch-main/save/bg_50_no_crop/gray_rt_288_snr_1_25_new_bg_longer_dimmer/single_stage_with_sep_and_ignore_border/vis/tile_fp/',
    # Example: "fp_labels.txt". Leave as None to use CLI -o.
    "output_path": "/home/ckchng/Documents/realtime-semantic-segmentation-pytorch-main/save/bg_50_no_crop/gray_rt_288_snr_1_25_new_bg_longer_dimmer/single_stage_with_sep_and_ignore_border/vis/fp_labels.txt",
    # Window title
    "window_name": "fp_labelling",
    # Figure size in inches (width, height)
    "figsize": (24, 18),
}


def read_output_status(output_path: Path) -> Tuple[Set[str], Optional[str]]:
    if not output_path.exists():
        return set(), None
    seen: Set[str] = set()
    last_name: Optional[str] = None
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            seen.add(parts[0])
            last_name = parts[0]
    return seen, last_name


def iter_images(input_dir: Path) -> List[Path]:
    files = [p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    return sorted(files, key=lambda p: p.name.lower())


def prompt_label() -> Optional[str]:
    while True:
        print("Label this image:")
        print("  1) not relevant")
        print("  2) streaks")
        print("  3) actual fp")
        print("  4) hard to categorize")
        print("  q) quit")
        choice = input("Enter choice (1-4 or q): ").strip().lower()
        if choice == "q":
            return None
        if choice in LABELS:
            return LABELS[choice]
        print("Invalid choice. Try again.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Iterate through images, display one at a time, and label."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        nargs="?",
        default=None,
        help="Directory containing images to label (optional if set in HARD_CONFIG).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("fp_labels.txt"),
        help="Output text file (default: fp_labels.txt).",
    )
    args = parser.parse_args()

    if HARD_CONFIG["input_dir"] is not None:
        input_dir = Path(HARD_CONFIG["input_dir"])
    else:
        input_dir = args.input_dir

    if HARD_CONFIG["output_path"]:
        output_path = Path(HARD_CONFIG["output_path"])
    else:
        output_path = args.output

    if input_dir is None:
        print("Input directory not provided. Set HARD_CONFIG['input_dir'] or pass it on the CLI.", file=sys.stderr)
        return 2

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Input directory not found: {input_dir}", file=sys.stderr)
        return 2

    images = iter_images(input_dir)
    if not images:
        print(f"No images found in {input_dir}", file=sys.stderr)
        return 1

    seen_names, last_name = read_output_status(output_path)
    if last_name:
        print(f"Resuming after last labeled: {last_name} ({len(seen_names)} labeled)")
    else:
        print("No previous labels found. Starting fresh.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.ion()
    fig, ax = plt.subplots(figsize=HARD_CONFIG["figsize"])
    manager = getattr(fig.canvas, "manager", None)
    if manager is not None and hasattr(manager, "set_window_title"):
        manager.set_window_title(HARD_CONFIG["window_name"])
    plt.show(block=False)

    with output_path.open("a", encoding="utf-8") as out_f:
        for img_path in images:
            try:
                img = mpimg.imread(str(img_path))
            except Exception:
                print(f"Skipping unreadable image: {img_path.name}")
                continue

            ax.clear()
            if getattr(img, "ndim", 0) == 2:
                ax.imshow(img, cmap="gray")
            else:
                ax.imshow(img)
            ax.set_axis_off()
            ax.set_title(img_path.name)
            fig.canvas.draw_idle()
            plt.pause(0.001)

            label = prompt_label()
            if label is None:
                break

            if img_path.name in seen_names:
                print(f"Already labeled: {img_path.name}")
                continue

            out_f.write(f"{img_path.name}\t{label}\n")
            out_f.flush()
            seen_names.add(img_path.name)
            print(f"Saved: {img_path.name} -> {label}")

    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
