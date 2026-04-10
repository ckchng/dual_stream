"""
Compute frequency of each class in a COCO-like annotation JSON.

Edit `INPUT_JSON` below to point to your annotation file, then run:

    python tools/labelling/class_frequency.py

The script prints a sorted table (most frequent first) and writes a JSON summary
alongside the input file named `<input>_class_freq.json`.
"""

import json
import os
from collections import Counter

# Hardcoded input path — edit this to your annotation JSON
INPUT_JSON = "/home/ckchng/Documents/SDA_ODA/LMA_data/testing_data_label_with_dual_single_stage_and_rt_two_stage_labeled_merged_labels.json"


def load_annotations(path):
    with open(path, 'r') as f:
        return json.load(f)


def compute_frequencies(data):
    annotations = data.get('annotations', [])
    categories = data.get('categories', [])

    # Build id -> name mapping from categories list (if present)
    id2name = {c['id']: c['name'] for c in categories if 'id' in c and 'name' in c}

    by_id = Counter()
    by_name = Counter()

    for a in annotations:
        cat_name = a.get('category')
        cat_id = a.get('category_id')

        if isinstance(cat_id, int):
            by_id[cat_id] += 1
            if cat_id in id2name:
                by_name[id2name[cat_id]] += 1
            elif isinstance(cat_name, str):
                by_name[cat_name] += 1
        elif isinstance(cat_name, str):
            by_name[cat_name] += 1
        else:
            # unknown / unlabeled
            by_name['<unknown>'] += 1

    # For any ids present in id2name but zero in by_id, include zero counts in by_name if needed
    for cid, name in id2name.items():
        if name not in by_name:
            by_name[name] += 0
        if cid not in by_id:
            by_id[cid] += 0

    return by_id, by_name


def save_summary(out_path, summary):
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)


def main():
    inp = INPUT_JSON
    if not os.path.isfile(inp):
        print(f"Input JSON not found: {inp}")
        return

    data = load_annotations(inp)
    by_id, by_name = compute_frequencies(data)

    total_annotations = sum(by_id.values()) if sum(by_id.values()) > 0 else sum(by_name.values())
    total_images = len(data.get('images', []))

    print(f"Total images: {total_images}")
    print(f"Total annotations: {total_annotations}\n")

    print("Top classes by name:")
    for name, cnt in by_name.most_common():
        print(f" - {name}: {cnt}")

    print('\nTop classes by id:')
    for cid, cnt in by_id.most_common():
        print(f" - {cid}: {cnt}")

    out_path = os.path.splitext(inp)[0] + '_class_freq.json'
    summary = {
        'input': inp,
        'total_images': total_images,
        'total_annotations': total_annotations,
        'by_id': dict(by_id),
        'by_name': dict(by_name)
    }

    save_summary(out_path, summary)
    print(f"\nWrote class frequency summary to: {out_path}")


if __name__ == '__main__':
    main()
