import numpy as np
import json

def line_angle_degrees(p1, p2, p3, p4):
    """
    p1, p2, p3, p4: (x, y) tuples or arrays
    Returns the (unsigned) angle between the two lines in degrees.
    """
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    p3 = np.array(p3, dtype=float)
    p4 = np.array(p4, dtype=float)

    v1 = p2 - p1
    v2 = p4 - p3

    # lengths
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        raise ValueError("One of the lines has zero length.")

    cos_theta_pos = np.dot(v1, v2) / (n1 * n2)
    # clamp to [-1, 1]
    cos_theta_pos = np.clip(cos_theta_pos, -1.0, 1.0)

    theta_pos = np.arccos(cos_theta_pos)          # in radians

    cos_theta_neg = np.dot(v1, -v2) / (n1 * n2)
    # clamp to [-1, 1]
    cos_theta_neg = np.clip(cos_theta_neg, -1.0, 1.0)

    theta_neg = np.arccos(cos_theta_neg)          # in radians
    return min(np.degrees(theta_pos), np.degrees(theta_neg))   

def line_intersection_check(pred_lines, gt_lines):
    # check if the box formed by each pred_line overlaps with any gt line. Collect all matches
    gt_box_matched_id = [[] for _ in range(len(gt_lines))]
    for pred_idx, pred_line in enumerate(pred_lines):
        x1_p, y1_p, x2_p, y2_p = pred_line
        pred_box = [min(x1_p, x2_p), min(y1_p, y2_p), max(x1_p, x2_p), max(y1_p, y2_p)]  # xmin, ymin, xmax, ymax

        matched = False
        for idx, line in enumerate(gt_lines):
            x1_g, y1_g, x2_g, y2_g = line
            gt_box = [min(x1_g, x2_g), min(y1_g, y2_g), max(x1_g, x2_g), max(y1_g, y2_g)]  # xmin, ymin, xmax, ymax

            # compute intersection
            ixmin = max(pred_box[0], gt_box[0])
            iymin = max(pred_box[1], gt_box[1])
            ixmax = min(pred_box[2], gt_box[2])
            iymax = min(pred_box[3], gt_box[3])
            iw = max(ixmax - ixmin + 1., 0.)
            ih = max(iymax - iymin + 1., 0.)
            inters = iw * ih

            if inters > 0:  # check for any overlap
                gt_box_matched_id[idx].append(pred_idx)
    return gt_box_matched_id

if __name__ == "__main__":
    stats_dir = '/home/ckchng/Documents/realtime-semantic-segmentation-pytorch-main/save/bg_50_no_crop/gray_rt_288_snr_1_25_new_bg_longer_dimmer/vis/full_frame_result/'
    # pred_dir ='/home/ckchng/Documents/realtime-semantic-segmentation-pytorch-main/save/bg_50_no_crop/gray_rt_288_snr_1_25_new_bg_longer_dimmer/vis/merged_text_ang5_dist10.0_eps0.01/txt/'
    pred_dir ='/home/ckchng/Documents/realtime-semantic-segmentation-pytorch-main/save/bg_50_no_crop/gray_rt_288_snr_1_25_new_bg_longer_dimmer/vis/full_frame_result/txt/'
    gt_dir = "/home/ckchng/Documents/SDA_ODA/LMA_data/testing_data_label.json"


    with open(gt_dir, 'r') as f:
        data_json = json.load(f)
        
    image_dirs = data_json['images']
    annotations = data_json['annotations']
    # comb_recall = []
    # comb_precision = []
    comb_tp = []
    comb_fn = []
    comb_fp = []
    comb_recall = []
    comb_precision = []
    tp_streak_len = []
    fn_streak_len = []
    for int_id in range(len(image_dirs)):
        curr_img_dir = image_dirs[int_id]['file_name']
        img_id = image_dirs[int_id]['id']

        # retrieve all annotation entries with the current img_id
        anno = [anno for anno in annotations if anno['image_id'] == image_dirs[int_id]['id']]
        gt_lines = [a['xyxy'] for a in anno]

        # read the corresponding pred in pred_dir
        pred_path = pred_dir + curr_img_dir.split('/')[-1].replace('.npy', '_lines.txt')
        pred_lines = []

        with open(pred_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    coords = list(map(float, line.split(',')))
                    pred_lines.append(coords)

        # check if the box formed by each pred_line overlaps with any gt line. Collect all matches
        gt_box_matched_id = line_intersection_check(pred_lines, gt_lines)
                    
        gt_line_matched_id = np.ones(len(gt_lines)) * -1
        # for each matched_gt_lines, measure the angular distance between gt_line and pred_line
        for gt_idx, matched_pred_indices in enumerate(gt_box_matched_id):

            if not matched_pred_indices:
                # fn_streak_len.append(gt_len)
                continue  # no match

            gt_line = gt_lines[gt_idx]
            gt_len = np.linalg.norm(np.array([gt_line[2]-gt_line[0], gt_line[3]-gt_line[1]]))


            # Check all candidate predictions
            for pred_idx in matched_pred_indices:
                pred_line = pred_lines[pred_idx]

                # compute angle of each line
                ang_diff = line_angle_degrees(gt_line[0:2], gt_line[2:4], pred_line[0:2], pred_line[2:4])

                # if ang_diff < 1 degree, a matched is found for the gt line
                if ang_diff < 10.0:
                    gt_line_matched_id[gt_idx] = pred_idx
                    tp_streak_len.append(gt_len)
                    break # Found a valid match for this GT line

        # iterate through gt_line_matched_id to find unmatched gt lines and record their lengths
        for gt_idx, matched_id in enumerate(gt_line_matched_id):
            if matched_id == -1:
                gt_line = gt_lines[gt_idx]
                gt_len = np.linalg.norm(np.array([gt_line[2]-gt_line[0], gt_line[3]-gt_line[1]]))
                fn_streak_len.append(gt_len)
                
        # compute recall, precision
        true_positives = np.sum(gt_line_matched_id != -1)
        false_negatives = np.sum(gt_line_matched_id == -1)
        false_positives = len(pred_lines) - true_positives

        # accumulate the lenghts of tp and fn based on gt line lenghts


        comb_tp.append(true_positives)
        comb_fn.append(false_negatives)
        comb_fp.append(false_positives)
        
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0

        comb_recall.append(recall)
        comb_precision.append(precision)
        # write in a text file the true_positives, false_negatives, false_positives, recall, precision along side the image name in the same line
        result_path=stats_dir + '/line_eval_stats.txt'
        with open(result_path, 'a') as f:
            f.write(f"Image: {curr_img_dir.split('/')[-1]}, TP: {true_positives}, FN: {false_negatives}, FP: {false_positives}, Recall: {recall:.4f}, Precision: {precision:.4f}\n")

    print("Overall Evaluation:")    
    print(f"Total True Positives: {sum(comb_tp)}")
    print(f"Total False Negatives: {sum(comb_fn)}")
    print(f"Total False Positives: {sum(comb_fp)}")
    overall_recall = sum(comb_tp) / (sum(comb_tp) + sum(comb_fn)) if (sum(comb_tp) + sum(comb_fn)) > 0 else 0.0
    overall_precision = sum(comb_tp) / (sum(comb_tp) + sum(comb_fp)) if (sum(comb_tp) + sum(comb_fp)) > 0 else 0.0
    print(f"Overall Recall: {overall_recall:.4f}")
    print(f"Overall Precision: {overall_precision:.4f}")    
    # print(np.mean(comb_recall))
    # print(np.mean(comb_precision))
    
    
    
    # plot the histogram of tp_streak_len and fn_streak_len
    import matplotlib.pyplot as plt
    # larger front size
    plt.rcParams.update({'font.size': 20})
    plt.hist(tp_streak_len, bins=100, alpha=0.5, label='True Positive Streak Lengths')
    plt.hist(fn_streak_len, bins=100, alpha=0.5, label='False Negative Streak Lengths')
    plt.xlabel('Streak Length')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Histogram of True Positive and False Negative Streak Lengths')
    plt.show()