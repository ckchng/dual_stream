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
    stats_dir = '/home/ckchng/Documents/realtime-semantic-segmentation-pytorch-main/save/bg_50_no_crop/gray_rt_288_snr_1_25_new_bg_longer_dimmer/single_stage_with_sep_and_ignore_border/vis/full_frame_result/'
    
    
    # pred_dir ='/home/ckchng/Documents/realtime-semantic-segmentation-pytorch-main/save/bg_50_no_crop/gray_rt_288_snr_1_25_new_bg_longer_dimmer/single_stage_with_sep_and_ignore_border/vis/full_frame_result/merged_txt_ang5_dist10.0_eps0.01/txt/'
    # pred_dir = '/home/ckchng/Documents/realtime-semantic-segmentation-pytorch-main/save/bg_50_no_crop/gray_rt_288_snr_1_25_new_bg_longer_dimmer/two_stage_with_sep_and_rc_ignore_border/vis/full_frame_result/merged_txt_ang5_dist10.0_eps0.01/txt/'
    pred_dir = '/home/ckchng/Documents/realtime-semantic-segmentation-pytorch-main/save/bg_50_no_crop/gray_rt_288_snr_1_25_new_bg_longer_dimmer/single_stage/vis/params_3.0_6_5.0_0.6_6.0_0.1/full_frame_result/merged_txt_ang5_dist10.0_eps0.01/txt/'
    pred_dir = '/home/ckchng/Documents/realtime-semantic-segmentation-pytorch-main/save/bg_50_no_crop/gray_rt_288_snr_1_25_new_bg_longer_dimmer/single_stage/vis/params_3.0_6_4.5_0.6_6.0_0.1/full_frame_result/merged_txt_ang5_dist10.0_eps0.01/txt/'
    pred_dir = '/home/ckchng/Documents/realtime-semantic-segmentation-pytorch-main/save/bg_50_no_crop/gray_rt_288_snr_1_25_new_bg_longer_dimmer/single_stage/vis/params_3.0_6_4.75_0.6_6.0_0.1/full_frame_result/merged_txt_ang5_dist10.0_eps0.01/txt/'
    pred_dir = '/home/ckchng/Documents/realtime-semantic-segmentation-pytorch-main/save/bg_50_no_crop/gray_rt_288_snr_1_25_new_bg_longer_dimmer/single_stage/vis/params_3.0_6_5.0_0.6_6.0_0.5/full_frame_result/merged_txt_ang5_dist10.0_eps0.01/txt/'
    pred_dir = '/home/ckchng/Documents/realtime-semantic-segmentation-pytorch-main/save/bg_50_no_crop/gray_rt_288_snr_1_25_new_bg_longer_dimmer/single_stage/vis/params_3.0_6_6.0_0.6_6.0_0.1/full_frame_result/merged_txt_ang5_dist10.0_eps0.01/txt/'
    pred_dir = '/home/ckchng/Documents/realtime-semantic-segmentation-pytorch-main/save/bg_50_no_crop/gray_rt_288_snr_1_25_new_bg_longer_dimmer/single_stage/vis/params_3.0_6_6.0_0.6_6.0_0.1/full_frame_result/merged_txt_ang5_dist10.0_eps0.01/txt/'
    pred_dir = '/home/ckchng/Documents/realtime-semantic-segmentation-pytorch-main/save/bg_50_no_crop/snr_1_25/two_classes/both/single_stage/vis/params_3.0_6_6.0_0.6_6.0_0.1/full_frame_result/merged_txt_ang5_dist10.0_eps0.01/txt/'
    pred_dir = '/home/ckchng/Documents/realtime-semantic-segmentation-pytorch-main/save/bg_50_no_crop/snr_1_25/two_classes/both_run2/single_stage/vis/params_3.0_6_6.0_0.6_6.0_0.1/full_frame_result/merged_txt_ang5_dist10.0_eps0.01/txt/'
    pred_dir = '/home/ckchng/Documents/dual_stream/dual_stream_two/save/bg_50_no_crop/snr_1_25_wo_borders/two_classes/both_run1/single_stage/vis/params_3.0_6_6.0_0.6_6.0_0.1/full_frame_result/merged_txt_ang5_dist10.0_eps0.01/txt/'
    pred_dir = '/home/ckchng/Documents/dual_stream/dual_stream_two/save/bg_50_no_crop/snr_1_25_wo_borders/two_classes/multi_model_eval/rt_run5/full_frame_result/merged_txt_ang5_dist10.0_eps0.01/txt/'
    # pred_dir = '/home/ckchng/Documents/realtime-semantic-segmentation-pytorch-main/save/bg_50_no_crop/snr_1_25_wo_borders/two_classes/both_new_mean/single_stage/vis/params_3.0_6_6.0_0.6_6.0_0.1/full_frame_result/merged_txt_ang5_dist10.0_eps0.01/txt/'
    
    # pred_dir = '/home/ckchng/Documents/realtime-semantic-segmentation-pytorch-main/save/bg_50_no_crop/snr_1_25_wo_borders/two_classes/rt_only/single_stage/vis/params_3.0_6_6.0_0.6_6.0_0.1/full_frame_result/merged_txt_ang5_dist10.0_eps0.01/txt/'
    # pred_dir = '/home/ckchng/Documents/realtime-semantic-segmentation-pytorch-main/save/bg_50_no_crop/gray_rt_288_snr_1_15_new_bg_longer_dimmer/rt_map_only/vis/full_frame_result/merged_txt_ang5_dist10.0_eps0.01/txt/'
    # pred_dir = '/home/ckchng/Documents/realtime-semantic-segmentation-pytorch-main/save/bg_50_no_crop/gray_rt_288_snr_1_15_new_bg_longer_dimmer/rt_map_only/two_stage_with_sep_rc_ignore_border/vis/full_frame_result/merged_txt_ang5_dist10.0_eps0.01/txt/'
    
    # gt_dir = "/home/ckchng/Documents/SDA_ODA/LMA_data/testing_data_label.json"
    # gt_dir = "/home/ckchng/Documents/SDA_ODA/LMA_data/testing_data_label.json"
    # gt_dir = '/home/ckchng/Documents/SDA_ODA/LMA_data/testing_data_label_with_dual_single_stage_labeled_with_dual_single_stage_and_rt_two_stage_labeled_merged_labels.json'
    gt_dir = '/home/ckchng/Documents/SDA_ODA/LMA_data/testing_data_label_with_dual_single_stage_and_rt_two_stage_labeled_merged_labels.json'



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
    # Counters for detection/miss per class (1, 9, 10)
    detect_counts = {1: 0, 9: 0, 10: 0}
    miss_counts = {1: 0, 9: 0, 10: 0}
    
    # Per-class detected/missed length lists
    detected_lengths = {1: [], 9: [], 10: []}
    missed_lengths = {1: [], 9: [], 10: []}
    missed_details = {1: [], 9: [], 10: []}
    for int_id in range(len(image_dirs)):
        curr_img_dir = image_dirs[int_id]['file_name']
        img_id = image_dirs[int_id]['id']

        # retrieve all annotation entries with the current img_id
        anno = [anno for anno in annotations if anno['image_id'] == image_dirs[int_id]['id']]
        gt_lines_all = [a['xyxy'] for a in anno]
        gt_cats_all = [a.get('category_id', None) for a in anno]

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
        gt_box_matched_id = line_intersection_check(pred_lines, gt_lines_all)

        # Build pred->gt mapping and initial pred statuses
        pred_count = len(pred_lines)
        pred_used = np.zeros(pred_count, dtype=bool)  # true if used as TP
        pred_status = ['unknown'] * pred_count  # 'unknown','candidate_tp','ignore','count5','count6','fp_candidate'
        count_cat5_preds = 0
        count_cat6_preds = 0
        count_cat1_preds = 0
        count_cat9_10_preds = 0

        # For each prediction, find which single GT it should associate with.
        # If a prediction overlaps multiple GTs, associate it with the GT that has the largest intersection area.
        pred_to_gt = {i: [] for i in range(pred_count)}
        pred_best_gt = {i: None for i in range(pred_count)}
        gt_to_preds_best = {i: [] for i in range(len(gt_lines_all))}

        # build boxes for preds and gts
        pred_boxes = []
        for pred_line in pred_lines:
            x1_p, y1_p, x2_p, y2_p = pred_line
            pred_boxes.append([min(x1_p, x2_p), min(y1_p, y2_p), max(x1_p, x2_p), max(y1_p, y2_p)])
        gt_boxes = []
        for gt_line in gt_lines_all:
            x1_g, y1_g, x2_g, y2_g = gt_line
            gt_boxes.append([min(x1_g, x2_g), min(y1_g, y2_g), max(x1_g, x2_g), max(y1_g, y2_g)])

        for pidx, pbox in enumerate(pred_boxes):
            best_gt = None
            best_area = 0.0
            for gidx, gbox in enumerate(gt_boxes):
                ixmin = max(pbox[0], gbox[0])
                iymin = max(pbox[1], gbox[1])
                ixmax = min(pbox[2], gbox[2])
                iymax = min(pbox[3], gbox[3])
                iw = max(ixmax - ixmin + 1., 0.)
                ih = max(iymax - iymin + 1., 0.)
                inters = iw * ih
                if inters > best_area:
                    best_area = inters
                    best_gt = gidx
            if best_area > 0:
                pred_best_gt[pidx] = best_gt
                pred_to_gt[pidx] = [best_gt]
                gt_to_preds_best[best_gt].append(pidx)
            else:
                pred_to_gt[pidx] = []

        # Now set statuses based on the single best GT (if any)
        for pidx in range(pred_count):
            gt_idxs = pred_to_gt.get(pidx, [])
            if not gt_idxs:
                pred_status[pidx] = 'fp_candidate'
                continue
            # we only associated to the best GT, so take that
            best_gt = gt_idxs[0]
            cat = gt_cats_all[best_gt]
            # If intersects with category 1, 9, or 10 -> candidate for TP (angle check will confirm)
            # create two different counts, one for category 1, and one for 9 and 10
            if cat in (1, 9, 10):
                pred_status[pidx] = 'candidate_tp'
                if cat == 1:
                    count_cat1_preds += 1
                else:                    
                    count_cat9_10_preds += 1   
                continue
            # Else if intersects with category 3 (irrelevant) -> discard prediction
            if cat == 3:
                pred_status[pidx] = 'irrelevant'
                continue
            # Else if intersects with 5 or 6 -> discard and count individually
            if cat == 5:
                pred_status[pidx] = 'hard_to_categorize'
                count_cat5_preds += 1
                continue
            # if cat == 6:
            #     pred_status[pidx] = 'count6'
            #     count_cat6_preds += 1
            #     continue
            # Otherwise (other categories), treat as false positive candidate
            pred_status[pidx] = 'fp_candidate'

        # Now perform angle-based matching but only for GT categories 1 and 3
        gt_line_matched_id = np.ones(len(gt_lines_all), dtype=int) * -1
        for gt_idx, matched_pred_indices in gt_to_preds_best.items():
            # Consider GT categories 1, 9 and 10 for angle-based matching
            if gt_cats_all[gt_idx] not in (1, 9, 10):
                continue

            if not matched_pred_indices:
                continue

            gt_line = gt_lines_all[gt_idx]
            gt_len = np.linalg.norm(np.array([gt_line[2]-gt_line[0], gt_line[3]-gt_line[1]]))

            # Check candidate predictions for this GT
            for pred_idx in matched_pred_indices:
                # Ignore predictions that were discarded (status ignore/count5/count6)
                if pred_status[pred_idx] in ('irrelevant', 'hard_to_categorize'):
                    continue

                pred_line = pred_lines[pred_idx]
                ang_diff = line_angle_degrees(gt_line[0:2], gt_line[2:4], pred_line[0:2], pred_line[2:4])
                if ang_diff < 10.0:
                    gt_line_matched_id[gt_idx] = pred_idx
                    tp_streak_len.append(gt_len)
                    # record per-class detected length
                    cat_here = gt_cats_all[gt_idx]
                    if cat_here in detected_lengths:
                        detected_lengths[cat_here].append(gt_len)
                    pred_used[pred_idx] = True
                    break

        # For computing false negatives and detection counts for categories 1, 9, 10
        for gt_idx, cat in enumerate(gt_cats_all):
            if cat not in (1, 9, 10):
                continue

            gt_line = gt_lines_all[gt_idx]
            gt_len = np.linalg.norm(np.array([gt_line[2]-gt_line[0], gt_line[3]-gt_line[1]]))

            # For categories 1, 9, 10 use angle-based matching result
            if gt_line_matched_id[gt_idx] != -1:
                detect_counts[cat] += 1
                if cat in detected_lengths:
                    detected_lengths[cat].append(gt_len)
            else:
                miss_counts[cat] += 1
                fn_streak_len.append(gt_len)
                if cat in missed_lengths:
                    missed_lengths[cat].append(gt_len)
                # record image name (basename) and gt index for later inspection
                img_name = curr_img_dir.split('/')[-1]
                if cat in missed_details:
                    missed_details[cat].append({'image': img_name, 'length': float(gt_len), 'gt_index': int(gt_idx)})


        # compute recall, precision
        true_positives = int(np.sum(gt_line_matched_id != -1))
        false_negatives = int(np.sum([(1 if (cat in (1,9,10) and gt_line_matched_id[idx] == -1) else 0) for idx, cat in enumerate(gt_cats_all)]))
        # false positives: predictions that were not used as TP and not discarded
        false_positives = 0
        for pidx in range(pred_count):
            if pred_used[pidx]:
                continue
            if pred_status[pidx] in ('irrelevant', 'hard_to_categorize'):
                continue
            false_positives += 1

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
    # Plot detection/missed ratio for classes 1,9,10
    classes = [1, 9, 10]
    detected = [detect_counts[c] for c in classes]
    missed = [miss_counts[c] for c in classes]
    totals = [detected[i] + missed[i] for i in range(len(classes))]
    detected_ratio = [detected[i] / totals[i] if totals[i] > 0 else 0.0 for i in range(len(classes))]
    missed_ratio = [missed[i] / totals[i] if totals[i] > 0 else 0.0 for i in range(len(classes))]

    x = np.arange(len(classes))
    width = 0.35
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    bars1 = ax2.bar(x - width/2, detected_ratio, width, label='Detected')
    bars2 = ax2.bar(x + width/2, missed_ratio, width, label='Missed')
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(c) for c in classes])
    ax2.set_ylim(0, 1)
    ax2.set_xlabel('Category')
    ax2.set_ylabel('Ratio')
    ax2.set_title('Detected vs Missed Ratio per Category (1,9,10)')
    ax2.legend()

    # annotate counts above bars
    for rect, cnt in zip(bars1, detected):
        height = rect.get_height()
        ax2.annotate(f'{cnt}', xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 3), textcoords='offset points', ha='center', va='bottom')
    for rect, cnt in zip(bars2, missed):
        height = rect.get_height()
        ax2.annotate(f'{cnt}', xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 3), textcoords='offset points', ha='center', va='bottom')

    plt.show()
    plt.hist(tp_streak_len, bins=100, alpha=0.5, label='True Positive Streak Lengths')
    plt.hist(fn_streak_len, bins=100, alpha=0.5, label='False Negative Streak Lengths')
    plt.xlabel('Streak Length')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Histogram of True Positive and False Negative Streak Lengths')
    plt.show()

    # Per-class length-based histograms (detected vs missed)
    fig3, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for i, c in enumerate(classes):
        ax = axes[i]
        det = detected_lengths.get(c, [])
        mis = missed_lengths.get(c, [])
        if det:
            ax.hist(det, bins=50, alpha=0.6, label='Detected')
        if mis:
            ax.hist(mis, bins=50, alpha=0.6, label='Missed')
        ax.set_title(f'Category {c} streak lengths')
        ax.set_xlabel('Streak Length')
        if i == 0:
            ax.set_ylabel('Frequency')
        ax.legend()
    plt.suptitle('Per-class Detected vs Missed Streak Lengths')
    plt.show()


    # Save false-negative details to a JSON file for inspection
    try:
        out_details_path = stats_dir + '/false_negative_details.json'
        with open(out_details_path, 'w') as _f:
            json.dump(missed_details, _f, indent=2)
        print(f'Wrote false-negative details to: {out_details_path}')
    except Exception as e:
        print(f'Failed to write false-negative details: {e}')