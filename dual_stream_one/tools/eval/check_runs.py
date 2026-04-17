"""
Check all runs under a given directory and report the best epoch,
mIoU_s1, and mIoU_s2 (if available) for each run.

Supports two log formats:
  Dual-stream:  Epoch<N> mIoU_s1: X.XXXX    | mIoU_s2: X.XXXX    | best ...
  Single-stream: Epoch<N> mIoU: X.XXXX    | best ...
"""
import os
import re
import sys

RUNS_DIR = "/home/ckchng/Documents/dual_stream/dual_stream_one/save/snr_1_32_len_200_for_m1/single_class/"

# Regex for dual-stream lines
RE_DUAL = re.compile(
    r"Epoch(\d+)\s+mIoU_s1:\s*([\d.]+)\s*\|\s*mIoU_s2:\s*([\d.]+)"
)
# Regex for single-stream lines
RE_SINGLE = re.compile(
    r"Epoch(\d+)\s+mIoU:\s*([\d.]+)"
)


def parse_log(log_path):
    """
    Returns a list of (epoch, miou_s1, miou_s2_or_None) tuples,
    one per epoch found in the log.
    Uses only the LAST occurrence of each epoch number (handles resumed runs).
    """
    epoch_data = {}  # epoch -> (miou_s1, miou_s2)

    with open(log_path, "r", errors="replace") as f:
        for line in f:
            line = line.strip()
            m = RE_DUAL.search(line)
            if m:
                epoch = int(m.group(1))
                s1 = float(m.group(2))
                s2 = float(m.group(3))
                epoch_data[epoch] = (s1, s2)
                continue

            m = RE_SINGLE.search(line)
            if m:
                epoch = int(m.group(1))
                s1 = float(m.group(2))
                epoch_data[epoch] = (s1, None)

    return epoch_data


def best_epoch(epoch_data):
    """Return (epoch, miou_s1, miou_s2) for the epoch with highest mIoU_s1."""
    if not epoch_data:
        return None, None, None
    best_ep = max(epoch_data, key=lambda e: epoch_data[e][0])
    s1, s2 = epoch_data[best_ep]
    return best_ep, s1, s2


def main():
    runs_dir = RUNS_DIR
    if len(sys.argv) > 1:
        runs_dir = sys.argv[1]

    runs = sorted(
        d for d in os.listdir(runs_dir)
        if os.path.isdir(os.path.join(runs_dir, d))
    )

    if not runs:
        print(f"No subdirectories found in {runs_dir}")
        return

    # Column widths
    col_run  = max(len("Run"), max(len(r) for r in runs))
    col_ep   = 12
    col_s1   = 12
    col_s2   = 12

    header = (
        f"{'Run':<{col_run}}  "
        f"{'Best Epoch':>{col_ep}}  "
        f"{'mIoU_s1':>{col_s1}}  "
        f"{'mIoU_s2':>{col_s2}}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    for run in runs:
        log_path = os.path.join(runs_dir, run, "seg_trainer.log")
        if not os.path.exists(log_path):
            print(f"{run:<{col_run}}  {'(no log)':>{col_ep}}  {'—':>{col_s1}}  {'—':>{col_s2}}")
            continue

        epoch_data = parse_log(log_path)
        if not epoch_data:
            print(f"{run:<{col_run}}  {'(empty)':>{col_ep}}  {'—':>{col_s1}}  {'—':>{col_s2}}")
            continue

        ep, s1, s2 = best_epoch(epoch_data)
        s1_str = f"{s1:.4f}" if s1 is not None else "—"
        s2_str = f"{s2:.4f}" if s2 is not None else "N/A"
        print(
            f"{run:<{col_run}}  "
            f"{ep:>{col_ep}}  "
            f"{s1_str:>{col_s1}}  "
            f"{s2_str:>{col_s2}}"
        )

    print(sep)


if __name__ == "__main__":
    main()
