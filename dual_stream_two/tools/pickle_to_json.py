import os
import json
import pickle
import numpy as np
from PIL import Image


def pkl_to_coco(
    pkl_path,
    json_path,
    images_root=None,
    category_name="object",
):
    """
    Convert a pickle file with keys 'imgPath' and 'XY' into COCO-format JSON.

    - pkl_path: path to the pickle file
    - json_path: where to save the COCO json
    - images_root: root directory of images (if imgPath is relative)
    - category_name: name of the single category used
    """
    # -------------------------
    # 1. Load pickle
    # -------------------------
    with open(pkl_path, "rb") as f:
        data_dict = pickle.load(f)

    if not isinstance(data_dict, dict):
        raise TypeError("Expected a dict in the pickle.")
    if "imgPath" not in data_dict or "XY" not in data_dict:
        raise KeyError("Missing required keys: imgPath, XY")

    # img_paths = list(data["imgPath"])  # list of strings
    # xy_list = list(data["XY"])         # entries like [[array()]], [[]], etc.

    print("Building NEF file mapping...")
    img_paths = {}
    for root, dirs, files in os.walk(images_root):
        for filename in files:
            if filename.endswith(".npy"):
                img_name = os.path.splitext(filename)[0]
                img_path = os.path.join(root, filename)
                img_paths[img_name] = img_path

    image_dirs = data_dict['imgPath']
    xy_list = data_dict['XY']
    # print(len(image_dirs))

    #for each image_dirs, find the corresponding in img_paths, and replace image_dirs with the path in image_paths
    for i in range(len(image_dirs)):
        img_name = os.path.splitext(os.path.basename(image_dirs[i]))[0]
        img_name = img_name.split('.')[0]  # in case there are multiple dots
        
        if img_name in img_paths:
            image_dirs[i] = img_paths[img_name]
        else:
            print(f"Image {img_name} not found in img_paths.")
    # -------------------------
    # 2. Flatten/pad/truncate XY the same way as your main()
    # -------------------------
# ---------- bbox(es) for this image ----------
    

    # -------------------------
    # 3. Prepare COCO dict
    # -------------------------
    coco = {
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": 1,
                "name": category_name,
                "supercategory": "none",
            }
        ]
    }

    ann_id = 1

    # -------------------------
    # 4. Fill images + annotations
    # -------------------------
    for img_id, (img_path, bbox_arr) in enumerate(zip(image_dirs, xy_list), start=1):
    
        img_file = img_path
    
        im = np.load(img_file)
        height, width = im.shape

        coco["images"].append(
            {
                "id": img_id,
                "file_name": img_path,
                "width": width,
                "height": height,
            }
        )

        arr = np.asarray(bbox_arr[0], dtype=float).ravel()

        # no annotations for this image -> keep image, skip annotations
        if arr.size == 0:
            continue

        # if not multiple of 4, something is off
        if arr.size % 4 != 0:
            print(
                f"Warning: XY for image {img_path} has {arr.size} values "
                f"(not divisible by 4). Skipping its boxes."
            )
            continue

        boxes = arr.reshape(-1, 4)  # (num_boxes, 4)

        for box in boxes:
            x1, y1, x2, y2 = box.tolist()
            coco["annotations"].append(
            {
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "xyxy": [x1, y1, x2, y2]
            }
            )
            ann_id += 1
 

        

        # Skip annotation if this was an empty / padded entry (NaNs)
        # if np.any(np.isnan(bbox_arr)):
        #     continue

        # # Interpret XY as [x_min, y_min, x_max, y_max]
        # x1, y1, x2, y2 = bbox_arr.tolist()
        # x = float(x1)
        # y = float(y1)
        # w = float(max(0.0, x2 - x1))
        # h = float(max(0.0, y2 - y1))

        # If your data already stores [x, y, w, h] instead, uncomment below:
        # x, y, w, h = bbox_arr.tolist()

        # area = float(w * h)



    # -------------------------
    # 5. Save JSON
    # -------------------------
    with open(json_path, "w") as f:
        json.dump(coco, f, indent=2)

    print(
        f"Saved COCO annotations for {len(coco['images'])} images "
        f"({len(coco['annotations'])} boxes). "
        # f"Padded/truncated entries (NaN -> skipped): {bad_idx}"
    )


if __name__ == "__main__":
    # Example usage:
    # pkl_to_coco("data.pkl", "annotations.json", images_root="path/to/images")
    # import argparse

    # parser = argparse.ArgumentParser()
    # parser.add_argument("pkl_path", type=str, help="Path to the input pickle file")
    # parser.add_argument("json_path", type=str, help="Path to output COCO JSON")
    # parser.add_argument(
    #     "--images_root",
    #     type=str,
    #     default=None,
    #     help="Root folder for images (if imgPath is relative)",
    # )
    # parser.add_argument(
    #     "--category_name",
    #     type=str,
    #     default="object",
    #     help="Category name for COCO annotations",
    # )
    # args = parser.parse_args()
    pkl_path = "/home/ckchng/Documents/SDA_ODA/LMA_data/testing_data_label.pkl"
    json_path = "/home/ckchng/Documents/SDA_ODA/LMA_data/testing_data_label.json"
    images_root = "/media/ckchng/internal2TB/FILTERED_IMAGES/"

    pkl_to_coco(
        pkl_path,
        json_path,
        images_root=images_root,
        category_name="object"
    )
