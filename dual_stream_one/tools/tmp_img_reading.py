import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

if __name__ == "__main__":
    img_dir = '/home/ckchng/Documents/SDA_ODA/LMA_data/gray_rt_288_snr_1_15_2_5_new_bg_longer_dimmer/labels/train/017_2020-12-08_135048_E_DSC_1648_x6450_y47_x6450_y47.png'
    # mask = cv2.imread(img_dir)
    # plt.imshow(mask * 255)
    # plt.show()
    # print('ck')
    data = np.load('/media/ckchng/internal2TB/FILTERED_IMAGES/FIREOPAL016/92000-110000/016_2020-12-08_105848_E_DSC_0616.npy')
    print('ck')
    # label_dir
    # label_dir = '/home/ckchng/Documents/SDA_ODA/LMA_data/testing_gray_rt_272_subset/text_labels/'
    # # mask_dir
    # mask_dir = '/home/ckchng/Documents/SDA_ODA/LMA_data/gray_rt_272_50_bg_mask/labels/val/rt_272/'
    # # iterate through label_dir, if the label text file is not emtpy, read the corresponding mask from mask_dir
    # all_labels = os.listdir(label_dir)
    # img_with_fg_count = 0
    # for label_file in all_labels:
    #     label_path = os.path.join(label_dir, label_file)
    #     with open(label_path, 'r') as f:
    #         lines = f.readlines()
    #         if len(lines) > 0:
    #             # read corresponding mask
    #             mask_file = label_file.replace('.txt', '.png')
    #             mask_path = os.path.join(mask_dir, mask_file)
    #             mask = cv2.imread(mask_path)
    #             mask = mask[:, :, 0]  # assume single channel
    #             unique_values = np.unique(mask)
    #             img_with_fg_count += 1
    #             if len(unique_values) < 2:
    #                 print(f"Mask {mask_file} does not have both classes. Unique values: {unique_values}")

    # print(f"Number of images with foreground: {img_with_fg_count}")
