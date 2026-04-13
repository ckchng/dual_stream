#!/usr/bin/env python
"""
summarise_logs.py
-----------------
Scans every subdirectory under a given root for seg_trainer.log and reports
the best validation mIoU (s1 and s2 if available) along with the epoch it
was achieved.

Usage:
    python summarise_logs.py /path/to/save/root
    python summarise_logs.py  # defaults to the path hard-coded in DEFAULT_ROOT
"""

import argparse
import os
import re
import sys

DEFAULT_ROOT = (
    "/home/ckchng/Documents/dual_stream/dual_stream_two/save/"
    "bg_50_no_crop/snr_1_25_wo_borders/two_classes"
)

# ── regex patterns ────────────────────────────────────────────────────────────
# Dual-stream line:  Epoch5 mIoU_s1: 0.8932    | mIoU_s2: 0.8395    | best ...
RE_DUAL = re.compile(
    r"Epoch(\d+)\s+mIoU_s1:\s*([\d.]+)\s*\|\s*mIoU_s2:\s*([\d.]+)"
)
# Single-stream line: Epoch5 mIoU: 0.8932    | best mIoU so far: ...
RE_SINGLE = re.compile(
    r"Epoch(\d+)\s+mIoU:\s*([\d.]+)"
)


def parse_log(log_path: str) -> dict:
    best_s1 = -1.0
    best_s2 = None
    best_epoch = -1
    is_dual = False

    with open(log_path, "r", errors="replace") as f:
        for line in f:
            m = RE_DUAL.search(line)
            if m:
                is_dual = True
                epoch = int(m.group(1))
                s1 = float(m.group(2))
                s2 = float(m.group(3))
                if s1 > best_s1:
                    best_s1 = s1
                    best_s2 = s2
                    best_epoch = epoch
                continue

            if not is_dual:
                m = RE_SINGLE.search(line)
                if m:
                    epoch = int(m.group(1))
                    s1 = float(m.group(2))
                    if s1 > best_s1:
                        best_s1 = s1
                        best_epoch = epoch

    return {
        "best_epoch": best_epoch,
        "best_miou_s1": best_s1 if best_s1 >= 0 else None,
        "best_miou_s2": best_s2,
        "is_dual": is_dual,
    }


def main():
    parser = argparse.ArgumentParser(description="Summarise seg_trainer.log files.")
    parser.add_argument(
        "root",
        nargs="?",
        default=DEFAULT_ROOT,
        help="Root directory to search (default: %(default)s)",
    )
    args = parser.parse_args()

    root = args.root
    if not os.path.isdir(root):
        print(f"ERROR: directory not found: {root}", file=sys.stderr)
        sys.exit(1)

    # Collect all logs, sorted by experiment name
    logs = []
    for entry in sorted(os.scandir(root), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        log_path = os.path.join(entry.path, "seg_trainer.log")
        if os.path.isfile(log_path):
            logs.append((entry.name, log_path))

    if not logs:
        print(f"No seg_trainer.log files found under: {root}")
        sys.exit(0)

    # ── header ────────────────────────────────────────────────────────────────
    col_exp    = 30
    col_epoch  = 7
    col_s1     = 12
    col_s2     = 12

    header = (
        f"{'Experiment':<{col_exp}} "
        f"{'Epoch':>{col_epoch}} "
        f"{'mIoU_s1':>{col_s1}} "
        f"{'mIoU_s2':>{col_s2}}"
    )
    sep = "-" * len(header)
    print(f"\nRoot: {root}\n")
    print(header)
    print(sep)

    for exp_name, log_path in logs:
        result = parse_log(log_path)

        s1_str = f"{result['best_miou_s1']:.4f}" if result["best_miou_s1"] is not None else "N/A"
        s2_str = f"{result['best_miou_s2']:.4f}" if result["best_miou_s2"] is not None else "N/A"
        epoch_str = str(result["best_epoch"]) if result["best_epoch"] >= 0 else "N/A"

        print(
            f"{exp_name:<{col_exp}} "
            f"{epoch_str:>{col_epoch}} "
            f"{s1_str:>{col_s1}} "
            f"{s2_str:>{col_s2}}"
        )

    print(sep)
    print(f"\n{len(logs)} experiment(s) found.\n")


if __name__ == "__main__":
    main()
