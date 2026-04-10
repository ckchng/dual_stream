import argparse, pickle, numpy as np


def main(pkl_path, npz_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, dict):
        raise TypeError("Expected a dict in the pickle.")
    if "imgPath" not in data or "XY" not in data:
        raise KeyError("Missing required keys: imgPath, XY")

    img_paths = list(data["imgPath"])  # list of strings
    xy_list = list(data["XY"])         # entries look like [[array()]], [[]], etc.

    xy_rows = []
    bad_idx = []
    for idx, item in enumerate(xy_list):
        arr = np.asarray(item, dtype=float).ravel()  # flatten nested lists/arrays

        if arr.size == 0:            # empty entry: pad with NaNs
            bad_idx.append(idx)
            arr = np.full(4, np.nan, dtype=float)
        elif arr.size < 4:           # shorter than 4: pad with NaNs
            padded = np.full(4, np.nan, dtype=float)
            padded[: arr.size] = arr
            arr = padded
        elif arr.size > 4:           # longer than 4: truncate
            arr = arr[:4]

        xy_rows.append(arr)

    xy_array = np.vstack(xy_rows) if xy_rows else np.empty((0, 4), dtype=float)

    # Save: paths as array of strings; xy as float array
    np.savez(npz_path, imgPath=np.asarray(img_paths, dtype=object), XY=xy_array)
    print(f"Saved {len(img_paths)} entries to {npz_path}. Padded/truncated entries: {bad_idx}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pickle_in", nargs="?", default="/home/ckchng/Documents/SDA_ODA/LMA_data/testing_data_label.pkl")
    parser.add_argument("npz_out", nargs="?", default="/home/ckchng/Documents/SDA_ODA/LMA_data/testing_data_label.npz")
    args = parser.parse_args()
    main(args.pickle_in, args.npz_out)
