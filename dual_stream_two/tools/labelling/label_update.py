import json
# insert a path for the merge_lines module, which is in the previous directory
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from merge_lines import merge_connected_segments_2d

if __name__ == "__main__":
    gt_dir = "/home/ckchng/Documents/SDA_ODA/LMA_data/testing_data_label.json"
    new_anno_dir_1 = "/home/ckchng/Documents/realtime-semantic-segmentation-pytorch-main/save/missed_streaks_from_detection/rt_raw_single_stage.txt"
    new_anno_dir_2 = "/home/ckchng/Documents/realtime-semantic-segmentation-pytorch-main/save/missed_streaks_from_detection/rt_two_stage.txt"
    fp_dir_1 = '/home/ckchng/Documents/realtime-semantic-segmentation-pytorch-main/save/bg_50_no_crop/gray_rt_288_snr_1_25_new_bg_longer_dimmer/single_stage_with_sep_and_ignore_border/vis/tile_fp/'
    fp_dir_2 = '/home/ckchng/Documents/realtime-semantic-segmentation-pytorch-main/save/bg_50_no_crop/gray_rt_288_snr_1_15_new_bg_longer_dimmer/rt_map_only/two_stage_with_sep_rc_ignore_border/vis/tile_fp/'
    with open(gt_dir, 'r') as f:
        data_json = json.load(f)
        
    image_dirs = data_json['images']
    annotations = data_json['annotations']
    # prepare categories mapping (name -> id); create categories list if missing
    categories = data_json.get('categories', [])
    name2cat = {c['name']: c['id'] for c in categories} if categories else {}
    next_cat_id = max([c['id'] for c in categories], default=0) + 1

    # extract all lines in new_anno_dir, each line is in format "img_name label"
    # example
    # 000_2020-12-08_095228_E_DSC_0219_y1584_x3456_pred0.png	streaks
    # 000_2020-12-08_095228_E_DSC_0219_y1728_x3168_pred0.png	streaks
    # 000_2020-12-08_095228_E_DSC_0219_y1872_x3024_pred0.png	streaks
    # 000_2020-12-08_095228_E_DSC_0219_y2736_x144_pred0.png	hard to categorize
    # 000_2020-12-08_095238_E_DSC_0220_y1152_x2592_pred0.png	not relevant
    # 000_2020-12-08_095238_E_DSC_0220_y1296_x2592_pred0.png	not relevant
    new_anno_dict_1 = []
    with open(new_anno_dir_1, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            img_name = parts[0]
            # retain only the part before "_y" in img_name
            # suffix it with ".png" to match the file_name in image_dirs
            pred_name = img_name
            img_name = img_name.split("_y")[0]
            img_name = img_name + ".npy"
            
            label = parts[1]
            
            new_anno_dict_1.append([img_name, pred_name, label])
    
    new_anno_dict_2 = []
    with open(new_anno_dir_2, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            img_name = parts[0]
            # retain only the part before "_y" in img_name
            # suffix it with ".png" to match the file_name in image_dirs
            pred_name = img_name
            img_name = img_name.split("_y")[0]
            img_name = img_name + ".npy"
            
            label = parts[1]
            
            new_anno_dict_2.append([img_name, pred_name, label])
    
    # iterate through the image_dirs
    for int_id in range(len(image_dirs)):
        curr_img_dir = image_dirs[int_id]['file_name']
        img_id = image_dirs[int_id]['id']

        # retrieve all annotation entries with the current img_id
        anno = [anno for anno in annotations if anno['image_id'] == image_dirs[int_id]['id']]
        gt_lines = [a['xyxy'] for a in anno]

        # collect all entries in new_anno_dict with the current img_name
        curr_new_anno_1 = [a for a in new_anno_dict_1 if a[0] == curr_img_dir.split('/')[-1]]
        curr_new_anno_2 = [a for a in new_anno_dict_2 if a[0] == curr_img_dir.split('/')[-1]]

        # read all the preds_line 
        pred_lines = []
        for entry in curr_new_anno_1:
            pred_name = entry[1]
            label = entry[2]
            curr_txt_path = fp_dir_1 + pred_name.replace('.png', '.txt')
            with open(curr_txt_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        coords = list(map(float, line.split(',')))
                        x1, y1, x2, y2 = coords
                        pred_lines.append([x1, y1, x2, y2, label])

        for entry in curr_new_anno_2:
            pred_name = entry[1]
            label = entry[2]
            curr_txt_path = fp_dir_2 + pred_name.replace('.png', '.txt')
            
            with open(curr_txt_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        coords = list(map(float, line.split(',')))
                        x1, y1, x2, y2 = coords
                        pred_lines.append([x1, y1, x2, y2, label])
            
            
        # merge the lines in curr_new_anno into the gt_lines, with the label as an additional field
        ma=5  # degrees
        md=10.0
        eps=1e-2
        
        merged_result = merge_connected_segments_2d(pred_lines, max_angle_deg=ma, max_dist_px=md, eps=eps)
        # merged_result is list of (segment, label)
        # for s, _ in merged_result:
        #     print(s)
        merged_segments = [[s[0][0], s[0][1], s[1][0], s[1][1]] for s, _ in merged_result]
        merged_labels = [l for _, l in merged_result]

        # retrieve the original gt_lines and append the merged_segments to it
        updated_gt_lines = gt_lines + merged_segments
        updated_gt_labels = ["original"] * len(gt_lines) + merged_labels

        # Prepare to append new annotations for the merged segments.
        # We'll update a copy of the original JSON: add categories if new labels are encountered,
        # and append new annotation entries with 'xyxy' and 'category_id'.
        # Build annotations list if not already handled
        # We'll modify data_json in-place (it's okay since we write out a new file later).
        if 'categories' not in data_json:
            data_json['categories'] = []
            categories = data_json['categories']

        # ensure annotations list exists
        if 'annotations' not in data_json:
            data_json['annotations'] = []
        new_annotations = data_json['annotations']

        # compute next annotation id
        try:
            max_ann_id = max(a.get('id', 0) for a in new_annotations)
        except ValueError:
            max_ann_id = 0
        next_ann_id = max_ann_id + 1

        # For each merged segment, create an annotation
        for seg, lab in zip(merged_segments, merged_labels):
            # seg is [x1,y1,x2,y2]
            x1, y1, x2, y2 = seg
            label_str = lab if (lab is not None and str(lab) != '') else 'merged_unknown'

            # map or create category
            if label_str in name2cat:
                cat_id = name2cat[label_str]
            else:
                cat_id = next_cat_id
                name2cat[label_str] = cat_id
                # append new category entry
                new_cat = {'id': cat_id, 'name': label_str}
                data_json['categories'].append(new_cat)
                next_cat_id += 1

            ann = {
                'id': next_ann_id,
                'image_id': img_id,
                'xyxy': [float(x1), float(y1), float(x2), float(y2)],
                'category_id': cat_id
            }
            new_annotations.append(ann)
            next_ann_id += 1

        print(f"Appended {len(merged_segments)} merged annotations for image id {img_id}")

    # After processing all images, write a new JSON file with merged annotations/categories
    out_json_path = gt_dir.replace('.json', '_with_dual_single_stage_and_rt_two_stage.json')
    try:
        with open(out_json_path, 'w') as out_f:
            json.dump(data_json, out_f)
        print(f"Wrote merged annotations JSON to {out_json_path}")
    except Exception as e:
        print(f"Error writing merged JSON: {e}")
