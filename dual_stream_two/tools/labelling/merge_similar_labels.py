"""
Merge similar/variant category names in a COCO-like annotation JSON.

Usage:
  python merge_similar_labels.py /path/to/annotations.json [--out /path/to/output.json]

The script will remap these variants by default:
  - 'obv streak' -> 'obs streak'
  - 'maybe streaks' -> 'maybe streak'

It updates annotation `category` (name) and `category_id` consistently, creates target categories if missing,
removes unused old categories, and writes a new JSON file.

It prints a short summary of how many annotations were remapped and which categories were removed.
"""

import json
import os
from collections import defaultdict


DEFAULT_MAPPING = {
    'obv streak': 'obs streak',
    'obv streaks': 'obs streak',
    'maybe streaks': 'maybe streak',
    'actual fp': 'obs fp',
}



def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def write_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def build_name_id_maps(categories):
    name2id = {}
    id2name = {}
    for c in categories:
        name = c.get('name')
        cid = c.get('id')
        if name is None or cid is None:
            continue
        name2id[name] = cid
        id2name[cid] = name
    return name2id, id2name


def merge_labels(data, mapping):
    categories = data.get('categories', [])
    annotations = data.get('annotations', [])

    name2id, id2name = build_name_id_maps(categories)
    max_id = max([c.get('id', 0) for c in categories], default=0)

    # prepare target ids (create if missing)
    created_cats = []
    for src, dst in mapping.items():
        if dst not in name2id:
            max_id += 1
            new_cat = {'id': max_id, 'name': dst}
            categories.append(new_cat)
            name2id[dst] = max_id
            id2name[max_id] = dst
            created_cats.append(dst)

    # track remap counts
    remap_counts = defaultdict(int)

    # for each annotation, remap by name or id
    for a in annotations:
        # prefer name if present
        cat_name = a.get('category')
        cat_id = a.get('category_id')

        # determine current name
        current_name = None
        if isinstance(cat_name, str):
            current_name = cat_name
        elif isinstance(cat_id, int):
            current_name = id2name.get(cat_id)

        if current_name is None:
            continue

        # if current name is in mapping, remap
        if current_name in mapping:
            new_name = mapping[current_name]
            new_id = name2id[new_name]
            a['category'] = new_name
            a['category_id'] = new_id
            remap_counts[(current_name, new_name)] += 1
        else:
            # also ensure category_id and name are consistent if one exists
            if isinstance(cat_name, str) and cat_name in name2id:
                a['category_id'] = name2id[cat_name]
            elif isinstance(cat_id, int) and cat_id in id2name:
                a['category'] = id2name[cat_id]

    # remove any categories that are old sources and no longer used
    used_category_ids = set()
    used_category_names = set()
    for a in annotations:
        if 'category_id' in a and isinstance(a['category_id'], int):
            used_category_ids.add(a['category_id'])
        if 'category' in a and isinstance(a['category'], str):
            used_category_names.add(a['category'])

    removed = []
    new_categories = []
    for c in categories:
        if c.get('id') in used_category_ids or c.get('name') in used_category_names:
            new_categories.append(c)
        else:
            removed.append(c.get('name'))

    data['categories'] = new_categories
    data['annotations'] = annotations

    return {
        'remap_counts': dict(remap_counts),
        'created_categories': created_cats,
        'removed_categories': removed,
    }


# Hardcoded input/output paths (edit these to point to your files)
INPUT_JSON = '/home/ckchng/Documents/SDA_ODA/LMA_data/testing_data_label_with_dual_single_stage_labeled.json'
OUT_JSON = None  # set to a path string to override default output


def main():
    inp = INPUT_JSON
    if not os.path.isfile(inp):
        print(f"Input file not found: {inp}")
        return

    data = load_json(inp)
    result = merge_labels(data, DEFAULT_MAPPING)

    out_path = OUT_JSON or (os.path.splitext(inp)[0] + '_merged_labels.json')
    write_json(data, out_path)

    print('\nMerge summary:')
    if result['remap_counts']:
        for (src, dst), cnt in result['remap_counts'].items():
            print(f" - {cnt} annotations: '{src}' -> '{dst}'")
    else:
        print(' - No annotations remapped.')

    if result['created_categories']:
        print('Created categories: ' + ', '.join(result['created_categories']))
    if result['removed_categories']:
        print('Removed categories: ' + ', '.join(result['removed_categories']))

    print(f"Wrote merged annotations to: {out_path}")


if __name__ == '__main__':
    main()
