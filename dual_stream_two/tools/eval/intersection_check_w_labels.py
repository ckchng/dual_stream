# take in a directory from a file to be compared
# iterate through
# extract the tile coordinates from the filename
# check if it exists in a label text file
# if it does, append it to a 'overlap' dict, with the category extracted from the label text file
# if it doesn't, append it to a 'non_overlap' dict
# for each tile in the label text file that has no corresponding tile in the directory, append it to a 'missed' dict
import json
import os
import re
# from merge_lines import merge_connected_segments_2d
from full_frame_line_eval import line_angle_degrees

if __name__ == "__main__":
    fp_dir = '/home/ckchng/Documents/realtime-semantic-segmentation-pytorch-main/save/bg_50_no_crop/gray_rt_288_snr_1_15_new_bg_longer_dimmer/rt_map_only/two_stage_with_sep_rc_ignore_border/vis/tile_fp/'
    label_txt_dir = '/home/ckchng/Documents/realtime-semantic-segmentation-pytorch-main/save/bg_50_no_crop/gray_rt_288_snr_1_25_new_bg_longer_dimmer/single_stage_with_sep_and_ignore_border/vis/fp_labels.txt'
    ref_fp_dir = '/home/ckchng/Documents/realtime-semantic-segmentation-pytorch-main/save/bg_50_no_crop/gray_rt_288_snr_1_25_new_bg_longer_dimmer/single_stage_with_sep_and_ignore_border/vis/tile_fp/'
    overlap_dict = {}
    non_overlap_dict = {}
    missed_dict = {}    
    # map fp image name -> matched label image name (so we can find ref txt files correctly)
    overlap_fp_to_labelimg = {}

    # iterate through the label text file and create a dict of img_name -> label
    label_dict = {}
    with open(label_txt_dir, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            img_name = parts[0]
            label = parts[1]
            label_dict[img_name] = label    

    # helper to parse coordinates from filenames like ..._y{y}_x{x}...
    coord_re = re.compile(r"_y(-?\d+)_x(-?\d+)")
    def parse_coords_from_name(name):
        # name can be with extension; strip it
        base = os.path.splitext(name)[0]
        m = coord_re.search(base)
        if not m:
            return None
        y = int(m.group(1))
        x = int(m.group(2))
        return x, y

    TILE_SIZE = 288

    def rect_from_xy(x, y, size=TILE_SIZE):
        # return (left, top, right, bottom)
        return (x, y, x + size, y + size)

    def rects_intersect(a, b):
        # a and b are (l,t,r,b)
        return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])

    fp_files = [f for f in os.listdir(fp_dir) if f.endswith('.txt')]
    label_img_names = list(label_dict.keys())

    # Pre-parse label coords for speed
    label_coords = {}
    for lab in label_img_names:
        c = parse_coords_from_name(lab)
        if c is not None:
            label_coords[lab] = rect_from_xy(*c)

    for fp_file in fp_files:
        img_name = fp_file.replace('.txt', '.png')
        fp_coords = parse_coords_from_name(img_name)
        if fp_coords is None:
            non_overlap_dict[img_name] = "no_label"
            continue
        fp_rect = rect_from_xy(*fp_coords)

        matched = False
        for lab_name, lab_rect in label_coords.items():
            if rects_intersect(fp_rect, lab_rect):
                overlap_dict[img_name] = label_dict.get(lab_name, "")
                overlap_fp_to_labelimg[img_name] = lab_name
                matched = True
                break
        if not matched:
            non_overlap_dict[img_name] = "no_label"

    # iterate through the label_dict and check if each label tile intersects any fp tile
    fp_rects = {}
    for fp_file in fp_files:
        fp_img = fp_file.replace('.txt', '.png')
        c = parse_coords_from_name(fp_img)
        if c is not None:
            fp_rects[fp_img] = rect_from_xy(*c)

    for lab_name in label_img_names:
        lab_rect = label_coords.get(lab_name)
        if lab_rect is None:
            # couldn't parse coords; treat as missed
            missed_dict[lab_name] = label_dict[lab_name]
            continue
        found = False
        for fp_img, fp_rect in fp_rects.items():
            if rects_intersect(lab_rect, fp_rect):
                found = True
                break
        if not found:
            missed_dict[lab_name] = label_dict[lab_name]

    # iterate through the overlap and read their corresponding txt files to check if their lines are actually close enough to be considered true positives
    # Iterate over a static list of keys so we can safely modify overlap_dict inside the loop.
    for img_name in list(overlap_dict.keys()):
        fp_file = img_name.replace('.png', '.txt')
        fp_path = os.path.join(fp_dir, fp_file)
        # Use the matched label image filename to build the reference txt path
        lab_img = overlap_fp_to_labelimg.get(img_name)
        if lab_img is not None:
            ref_file = lab_img.replace('.png', '.txt')
        else:
            ref_file = fp_file
        ref_path = os.path.join(ref_fp_dir, ref_file)
        # retrieve all lines in the ref_path and fp_path

        # then check if each line in fp_path is close enough to any line in ref_path, if it is, consider it a valid overlap, otherwise, consider it as a non-overlap and move it to the non_overlap_dict with the same label as in the overlap_dict, and remove it from the overlap_dict
        with open(fp_path, 'r') as f:
            fp_lines = [line.strip() for line in f if line.strip()]
        with open(ref_path, 'r') as f:
            ref_lines = [line.strip() for line in f if line.strip()]
        
        for fp_line in fp_lines:
            fp_coords = list(map(float, fp_line.split(',')))
            
            found_match = False
            for ref_line in ref_lines:
                ref_coords = list(map(float, ref_line.split(',')))
            
                ang_diff = line_angle_degrees(fp_coords[0:2], fp_coords[2:4], ref_coords[0:2], ref_coords[2:4])
                if ang_diff < 10.0: # if the angle difference is less than 5 degrees, consider it a match
                    found_match = True
                    break
            if not found_match:
                non_overlap_dict[img_name] = overlap_dict[img_name]
                # clean up mapping too
                if img_name in overlap_fp_to_labelimg:
                    del overlap_fp_to_labelimg[img_name]
                del overlap_dict[img_name]
                break

    # plot a histogram of the labels in the overlap_dict, non_overlap_dict and missed_dict
    import matplotlib.pyplot as plt
    from collections import Counter
    overlap_labels = list(overlap_dict.values())
    non_overlap_labels = list(non_overlap_dict.values())
    missed_labels = list(missed_dict.values())
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title("Overlap")
    plt.bar(Counter(overlap_labels).keys(), Counter(overlap_labels).values())
    plt.subplot(1, 3, 2)
    plt.title("Non-Overlap")
    plt.bar(Counter(non_overlap_labels).keys(), Counter(non_overlap_labels).values())
    plt.subplot(1, 3, 3)
    plt.title("Missed")
    plt.bar(Counter(missed_labels).keys(), Counter(missed_labels).values())
    plt.show() 
    
    # In overlap dicts, write in a text file the img_name and label of the streak:
    # with open(fp_dir + "overlap_labels.txt", "w") as f:
    #     for img_name, label in overlap_dict.items():
    #         f.write(f"{img_name}\t{label}\n")

    # # In non-overlap dicts, write in a text file the img_name and label of the streak:
    # with open(fp_dir + "non_overlap_labels.txt", "w") as f:
    #     for img_name, label in non_overlap_dict.items():
    #         f.write(f"{img_name}\t{label}\n")

    # # In missed dicts, write in a text file the img_name and label of the streak:
    # with open(fp_dir + "missed_labels.txt", "w") as f:
    #     for img_name, label in missed_dict.items():
    #         f.write(f"{img_name}\t{label}\n")