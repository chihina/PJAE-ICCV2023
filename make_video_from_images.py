import cv2
import os
import glob
import sys
import argparse

parser = argparse.ArgumentParser(description="parameters for training")
parser.add_argument("dataset", type=str, help="dataset name")
args = parser.parse_args()
dataset = args.dataset
dataset_list = ['volleyball', 'videocoatt']
for dataset in dataset_list:
    id_list = []
    if dataset == 'volleyball':
        model_name = 'volleyball-dual-mid_p_p_field_middle_p_s_davt_bbox_GT_gaze_GT_act_GT_weight_fusion_fine_token_only'
        id_list.append('4_105655')
        id_list.append('5_30480')
        # id_list.append('9_19275')
        # id_list.append('11_22120')
        # id_list.append('14_28045')
        # id_list.append('20_25385')
        # id_list.append('25_29630')
        # id_list.append('29_17050')
        id_list.append('34_12470')
    elif dataset == 'videocoatt':
        model_name = 'videocoatt-p_p_field_deep_p_s_davt_scalar_weight_fix_token_only_GT'
        id_list.append('10')
        id_list.append('15')
        id_list.append('19')
        id_list.append('23')

    for id_txt in id_list:
        print(f'{dataset}:{id_txt}')
        if dataset == 'volleyball':
            vid_num, seq_num = id_txt.split('_')
        elif dataset == 'videocoatt':
            vid_num = id_txt

        image_type_list = ['final_jo_att_superimposed']
        for image_type in image_type_list:
            video_folder = os.path.join('results', dataset, model_name, 'videos')
            if os.path.exists(video_folder) is False:
                os.makedirs(video_folder)

            image_folder = os.path.join('results', dataset, model_name, image_type)
            if dataset == 'volleyball':
                fps = 7
                images_file_base = os.path.join(image_folder, f'test_{vid_num}_{seq_num}_*_{image_type}.png')
            elif dataset == 'videocoatt':
                fps = 7
                images_file_base = os.path.join(image_folder, f'test_*_{vid_num}_*_{vid_num}_{image_type}.png')

            images = sorted(glob.glob(images_file_base))
            if dataset == 'volleyball':
                images_all = [images]
            elif dataset == 'videocoatt':
                images_all = []
                images_mini = []
                for img_idx, image in enumerate(images):
                    img_num = int(image.split('/')[-1].split('_')[3])

                    if img_idx != 0:
                        continue_flag = (prev_img_num+1) == img_num
                    else:
                        continue_flag = True

                    if continue_flag:
                        images_mini.append(image)
                    else:
                        images_all.append(images_mini)
                        images_mini = []

                    prev_img_num = img_num
                images_all.append(images_mini)

            for vid_idx, images in enumerate(images_all):
                frame = cv2.imread(images[0])
                height, width, layers = frame.shape
                
                if dataset == 'volleyball':
                    video_name = os.path.join(video_folder, f'test_{vid_num}_{seq_num}_{vid_idx}_{image_type}.avi')
                elif dataset == 'videocoatt':
                    video_name = os.path.join(video_folder, f'test_{vid_num}_{vid_idx}_{image_type}.avi')

                video = cv2.VideoWriter(video_name, 0, fps, (width,height))
                for image in images:
                    video.write(cv2.imread(image))
                cv2.destroyAllWindows()
                video.release()