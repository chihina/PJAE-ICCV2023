import cv2
import os
import glob
import sys
import argparse
from PIL import Image
from tqdm import tqdm

gif_img_vol = []
gif_img_vid = []
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
                gif_img_vol += images
            elif dataset == 'videocoatt':
                gif_img_vid += images

# pil_img_vol = [Image.open(gif_img).resize((320, 180)) for gif_img in gif_img_vol]
# pil_img_vid = [Image.open(gif_img).resize((320, 180)) for gif_img in gif_img_vid]
# pil_img_all = pil_img_vol + pil_img_vid
# pil_img_vol[0].save('results/results_vol.gif', save_all=True, append_images=pil_img_vol)
# pil_img_vid[0].save('results/results_vid.gif', save_all=True, append_images=pil_img_vid)
# pil_img_all[0].save('results/results_all.gif', save_all=True, append_images=pil_img_all)

pil_img_vol = [Image.open(gif_img).resize((320*2, 180*2)) for gif_img in gif_img_vol]
pil_img_vid = [Image.open(gif_img).resize((320*2, 180*2)) for gif_img in gif_img_vid]
pil_img_all = pil_img_vol + pil_img_vid
pil_img_vol[0].save('results/results_vol_large.gif', save_all=True, append_images=pil_img_vol, loop=0)
pil_img_vid[0].save('results/results_vid_large.gif', save_all=True, append_images=pil_img_vid, loop=0)
pil_img_all[0].save('results/results_all_large.gif', save_all=True, append_images=pil_img_all, loop=0)

# gif_img_all = gif_img_vol + gif_img_vid
# resize_height, resize_width = 720//4, 1280//4
# video_name = os.path.join('results', f'ICCV2023-PJAE-demo.mp4')
# fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
# video = cv2.VideoWriter(video_name, fmt, fps, (resize_width,resize_height))
# fps = 7
# for gif_img in tqdm(gif_img_all):
#     frame = cv2.resize(cv2.imread(gif_img), (resize_width,resize_height))
#     video.write(frame)
# cv2.destroyAllWindows()
# video.release()