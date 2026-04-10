# read the new anno_dir,
# retrieve the category,
# if the category is merged_unknown
# read the image, crop the merged_unknown region, prompt the user to label the region with a class
# the classes are 'not relevant, obs streak, maybe streak, obs fp, maybe fp, hard to categorized.
# update the category of the merged_unknown region in the annotation file to the user label
# save the final annotation file as a new json

import json
import os
import numpy as np
import matplotlib.pyplot as plt


def _ensure_int_coords(xyxy, img_shape):
    # Use min/max to get correct ordering and clamp to image bounds
    h, w = img_shape[0], img_shape[1]
    x1 = int(round(min(xyxy[0], xyxy[2])))
    x2 = int(round(max(xyxy[0], xyxy[2])))
    y1 = int(round(min(xyxy[1], xyxy[3])))
    y2 = int(round(max(xyxy[1], xyxy[3])))

    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))

    # ensure at least 1 pixel in each dimension
    if x2 <= x1:
        x2 = min(x1 + 1, w)
    if y2 <= y1:
        y2 = min(y1 + 1, h)
    return x1, y1, x2, y2


def build_img_paths(img_dir):
    img_paths = {}
    if not img_dir:
        return img_paths
    print("Building NEF/NPY file mapping...")
    for root, dirs, files in os.walk(img_dir):
        for filename in files:
            if filename.endswith('.npy'):
                img_name = os.path.splitext(filename)[0]
                img_path = os.path.join(root, filename)
                img_paths[img_name] = img_path
    return img_paths


def prompt_label_and_update(a, img, img_name, categories, name2cat, next_cat_id):
    # get bounding box
    if 'xyxy' in a:
        xyxy = a['xyxy']
    elif 'bbox' in a:
        # assume bbox is [x,y,w,h]
        x, y, w, h = a['bbox']
        xyxy = [x, y, x + w, y + h]
    else:
        print(f"Annotation for image {img_name} has no bbox/xyxy, skipping.")
        return next_cat_id

    x1, y1, x2, y2 = _ensure_int_coords(xyxy, img.shape)
    if x2 <= x1 or y2 <= y1:
        print(f"Invalid crop coords for {img_name}: {x1,y1,x2,y2}")
        return next_cat_id

    crop = img[y1:y2, x1:x2]

    zero_mask = (crop == 0)
    crop = crop - crop.min()
    crop = crop / (crop.max() + 1e-8) * 255
    crop[zero_mask] = 0
    
    plt.ion()
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    if crop.ndim == 2:
        ax.imshow(crop, cmap='gray')
    else:
        ax.imshow(crop)
    ax.axis('off')
    ax.set_title(f"Image: {img_name} — crop {x1},{y1},{x2},{y2}")
    plt.show(block=False)

    print(f"Please label the shown region in image {img_name}:")
    print("Classes: 1. not relevant, 2. obs streak, 3. maybe streak, 4. obs fp, 5. maybe fp, 6. hard to categorize")
    label_map = {
        '1': 'not relevant',
        '2': 'obs streak',
        '3': 'maybe streak',
        '4': 'obs fp',
        '5': 'maybe fp',
        '6': 'hard to categorize'
    }

    label_name = None
    try:
        while label_name is None:
            label = input("Enter the class number (or q to skip): ")
            if label.lower() == 'q':
                print('Skipping this annotation')
                return next_cat_id
            label_name = label_map.get(label, None)
            if label_name is None:
                print("Invalid label number. Try again or press q to skip.")
    finally:
        try:
            plt.close(fig)
        except Exception:
            pass
        plt.ioff()

    # ensure category exists
    if label_name not in name2cat:
        cat = {'id': next_cat_id, 'name': label_name}
        categories.append(cat)
        name2cat[label_name] = next_cat_id
        next_cat_id += 1

    # update annotation fields (support both name and id fields)
    cat_id = name2cat[label_name]
    a['category'] = label_name
    a['category_id'] = cat_id
    return next_cat_id


if __name__ == "__main__":
    gt_dir = "/home/ckchng/Documents/SDA_ODA/LMA_data/testing_data_label_with_rt_two_stage.json"
    gt_dir = '/home/ckchng/Documents/SDA_ODA/LMA_data/testing_data_label_with_dual_single_stage_and_rt_two_stage.json'
    img_dir = '/media/ckchng/internal2TB/FILTERED_IMAGES/'

    img_paths = build_img_paths(img_dir)

    with open(gt_dir, 'r') as f:
        data_json = json.load(f)

    image_dirs = data_json.get('images', [])
    annotations = data_json.get('annotations', [])
    categories = data_json.get('categories', [])

    name2cat = {c['name']: c['id'] for c in categories} if categories else {}
    next_cat_id = max([c.get('id', 0) for c in categories], default=0) + 1

    # find annotations that are to be determined
    tbd_annos = []
    tbd_id = name2cat.get('to_be_determined', None)
    for a in annotations:
        cat_field = a.get('category', a.get('category_id', None))
        if isinstance(cat_field, str) and cat_field == 'to_be_determined':
            tbd_annos.append(a)
        elif isinstance(cat_field, int) and tbd_id is not None and cat_field == tbd_id:
            tbd_annos.append(a)

    if not tbd_annos:
        print('No annotations labelled "to_be_determined" found.')
    else:
        # map image id -> file_name
        id2file = {img['id']: img['file_name'] for img in image_dirs}
        # process each annotation
        for a in tbd_annos:
            img_id = a['image_id']
            img_path = id2file.get(img_id)
            if img_path is None:
                print(f"Image id {img_id} not found in images list, skipping.")
                continue

            # base = os.path.splitext(img_name)[0]
            # img_path = img_paths.get(base)
            # if img_path is None:
            #     print(f"Image {img_name} not found in image directory mapping, skipping.")
            #     continue

            try:
                img = np.load(img_path)
            except Exception as e:
                print(f"Failed to load {img_path}: {e}")
                continue

            next_cat_id = prompt_label_and_update(a, img, os.path.splitext(img_path)[0], categories, name2cat, next_cat_id)

    # write out updated categories and annotations to new json
    data_json['categories'] = categories
    data_json['annotations'] = annotations
    out_path = os.path.splitext(gt_dir)[0] + '_labeled.json'
    with open(out_path, 'w') as f:
        json.dump(data_json, f, indent=2)
    print(f'Wrote updated annotations to {out_path}')