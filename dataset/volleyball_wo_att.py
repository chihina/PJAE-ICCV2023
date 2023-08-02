import torch
from torch.utils.data import Dataset
from torchvision import transforms

import numpy as np
import os
import sys
from PIL import Image
from tqdm import tqdm
import pandas as pd
import json

class VolleyBallDatasetWithoutAtt(Dataset):
    def __init__(self, cfg, mode, transform=None, pre_transform=None):

        # data
        self.sendo_dataset_dir = cfg.data.sendo_dataset_dir
        self.dataset_bbox_gt = cfg.data.dataset_bbox_gt
        self.dataset_bbox_pred = cfg.data.dataset_bbox_pred
        self.rgb_dataset_dir = cfg.data.rgb_dataset_dir
        self.annotation_dir = cfg.data.annotation_dir
        self.att_inside_dir = cfg.data.att_inside_dir
        self.data_name = cfg.data.name

        # exp settings
        self.wandb_name = cfg.exp_set.wandb_name
        self.model_type = cfg.model_params.model_type
        self.mode = mode
        self.resize_width = cfg.exp_set.resize_width
        self.resize_height = cfg.exp_set.resize_height
        self.resize_head_height = cfg.exp_set.resize_head_height
        self.resize_head_width = cfg.exp_set.resize_head_width
        self.resize_height_person = cfg.exp_set.resize_height_person
        self.resize_width_person = cfg.exp_set.resize_width_person

        # exp params
        self.use_frame_type = cfg.exp_params.use_frame_type
        self.bbox_types = cfg.exp_params.bbox_types
        self.gaussian_sigma_head = cfg.exp_params.gaussian_sigma
        self.pass_winpoint = True
        self.use_blured_img = cfg.exp_params.use_blured_img

        # data pack list
        self.gt_bbox = []
        self.gt_bbox_id = []
        self.gt_bbox_resized = []
        self.rgb_path_list = []
        self.person_bbox_dic = {}

        self.train_video = [1, 3, 6, 7, 10, 13, 15, 16, 18, 22,
                                   23, 31, 36, 38, 39, 40, 41, 42, 48,
                                   50, 52, 53, 54]
        self.valid_video = [0, 2, 8, 12, 17, 19, 24, 26, 27, 28,
                            30, 33, 46, 49, 51]
        self.test_video = [4, 5, 9, 11, 14, 20, 21, 25, 29, 34,
                           35, 37, 43, 44, 45, 47]

        if self.wandb_name == 'debug':
            self.train_video = [1, 3, 6]
            self.valid_video = [0, 2, 8]
            self.test_video = [4, 5, 9]

        if mode == 'train':
            self.use_video_list = self.train_video
        elif mode == 'valid':
            self.use_video_list = self.valid_video
        elif mode == 'test':
            self.use_video_list = self.test_video

        # make dataset
        self.generate_dataset_list()

        # set maximum number of people
        self.max_num_people = 0
        self.set_max_num_people()

        if self.use_blured_img:
            self.transforms_rgb = transforms.Compose(
                [
                    transforms.Resize((self.resize_height, self.resize_width)),
                    transforms.ToTensor(),
                    transforms.GaussianBlur(5, sigma=(2.0, 2.0)),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            self.transforms_rgb = transforms.Compose(
                [
                    transforms.Resize((self.resize_height, self.resize_width)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

        self.transforms_rgb_wo_norm = transforms.Compose(
            [
                transforms.Resize((self.resize_height, self.resize_width)),
                transforms.Resize((720, 1280)),
                transforms.ToTensor(),
            ]
        )

        self.transforms_gt = transforms.Compose(
            [
                transforms.Resize((self.resize_height, self.resize_width)),
            ]
        )

        self.transforms_person = transforms.Compose(
            [
                transforms.Resize((self.resize_height_person, self.resize_width_person)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    # generator gt gaussian
    def generate_2d_gaussian(self, height, width, peak_xy, sigma=1):
        peak_x, peak_y = peak_xy
        x = np.arange(0, width, 1)
        y = np.arange(0, height, 1)
        X,Y = np.meshgrid(x,y)
        heatmap = np.exp(- ((X-peak_x) ** 2 + (Y-peak_y) ** 2) / (2 * sigma ** 2))

        return heatmap.reshape(1, heatmap.shape[0], heatmap.shape[1])

    # get group activity recognition and individual action recognition data
    def get_gar_iar_ann_dic(self, gar_iar_ann_path):
        with open(gar_iar_ann_path, 'r') as f:
            gar_iar_lines = f.readlines()
        
        gar_iar_ann_dic = {}
        for gar_iar_line in gar_iar_lines:
            gar_iar_ann_dic[gar_iar_line.strip().split()[0].split('.')[0]] = gar_iar_line.strip()

        return gar_iar_ann_dic

    # set maximum number of people
    def set_max_num_people(self):
        for annotated_bbox in self.person_bbox_dic.values():
            for annotated_bbox_frame in annotated_bbox.values():
                data_people_num = len(annotated_bbox_frame.keys())
                self.max_num_people = max(self.max_num_people, data_people_num)

    def generate_dataset_list(self):
        for video_num in tqdm(sorted(self.use_video_list)):
            gar_iar_ann_path = os.path.join(self.rgb_dataset_dir, str(video_num), 'annotations.txt')
            gar_iar_ann_dic = self.get_gar_iar_ann_dic(gar_iar_ann_path)
            seq_cnt = 0
            for seq_num in sorted(os.listdir(os.path.join(self.rgb_dataset_dir, str(video_num)))):
                if 'txt' in seq_num:
                    continue

                gar_iar_ann_line = gar_iar_ann_dic[seq_num]
                gar_label = gar_iar_ann_line.split()[1]
                seq_cnt += 1
                
                if self.pass_winpoint and 'winpoint' in gar_label:
                    continue
                else:
                    pass

                if self.bbox_types == 'GT':
                    self.dataset_dir = self.dataset_bbox_gt
                elif self.bbox_types == 'PRED':
                    self.dataset_dir = self.dataset_bbox_pred
                else:
                    print('Employ correct bbox dataset')
                    sys.exit()                           

                # get gt person bbox
                annotation_path_person = os.path.join(self.sendo_dataset_dir, str(video_num), str(seq_num), f'{seq_num}.txt')
                annotated_bbox = self.get_person_bbox_from_txt(annotation_path_person)
                self.person_bbox_dic[f'{video_num}_{seq_num}'] = annotated_bbox

                # get gt ball bbox
                annotation_path = os.path.join(self.annotation_dir, f'volleyball_{video_num}_{seq_num}_ver3.csv')
                bbox_array, bbox_flag_array = self.read_ball_bbox_from_csv(annotation_path)

                if self.use_frame_type == 'mid':
                    img_num_list = [seq_num]
                    # print('=== Employ mid frames ===')
                elif self.use_frame_type == 'all':
                    img_num_list = sorted(annotated_bbox.keys())
                    # print('=== Employ all frames ===')
                else:
                    print('please select correct frame type')
                    sys.exit()
                
                for img_num in img_num_list:
                    frame_id = (int(img_num) - int(seq_num)) + 20

                    # get rgb file path
                    rgb_img_file_path = os.path.join(self.rgb_dataset_dir, str(video_num), str(seq_num), f'{img_num}.jpg')
                    img_width, img_height = Image.open(rgb_img_file_path).size

                    # if annotation dataset is nothing
                    if bbox_array.shape[0] == 0:
                        continue
                    ball_bbox = self.get_ball_bbox_from_csv(bbox_array, frame_id)
                    if bbox_flag_array[frame_id, 0] == 1:
                        continue

                    self.gt_bbox.append(ball_bbox)

                    x_min, y_min, x_max, y_max = map(int, ball_bbox)
                    x_min, x_max = map(lambda x: int(x*self.resize_width/img_width), [x_min, x_max])
                    y_min, y_max = map(lambda x: int(x*self.resize_height/img_height), [y_min, y_max])
                    gt_box = np.array([x_min, y_min, x_max, y_max])
                    gt_bbox_idx = np.arange(len(annotated_bbox[img_num].keys())).reshape(-1, 1)

                    self.gt_bbox_id.append(gt_bbox_idx)
                    self.gt_bbox_resized.append(gt_box)
                    self.rgb_path_list.append(rgb_img_file_path)

                # one seq in demo mode
                if self.wandb_name == 'debug' and seq_cnt > 10:
                    break
                if self.wandb_name == 'demo' and seq_cnt > 3:
                    break

    def __len__(self):
        return len(self.rgb_path_list)

    def __getitem__(self, idx):

        img_file_path = self.rgb_path_list[idx]
        img = Image.open(img_file_path)
        img_width, img_height = img.size

        bbox = self.gt_bbox[idx]
        gt_box = np.array(bbox)
        img_gt = self.load_gt_imgs_ball(img_width, img_height, bbox, self.gaussian_sigma_head)
        x_min, y_min, x_max, y_max = map(int, gt_box)
        x_mid, y_mid = (x_min+x_max)/2, (y_min+y_max)/2
        gt_box_expand = gt_box[None, :]
        gt_box_expand = np.tile(gt_box_expand, (self.max_num_people, 1))
        gt_box_expand[:, ::2] /= img_width
        gt_box_expand[:, 1::2] /= img_height

        head_img = torch.zeros(self.max_num_people, 3, self.resize_head_height, self.resize_head_width)
        head_vector_gt_tensor = torch.zeros(self.max_num_people, 2)
        head_feature_tensor = torch.zeros(self.max_num_people, 2+9)
        head_bbox_tensor = torch.zeros(self.max_num_people, 4)
        att_inside_flag = torch.zeros(self.max_num_people, dtype=torch.bool)
        img_person_tensor = torch.zeros(self.max_num_people, 3, self.resize_height_person, self.resize_width_person)
        people_bbox_tensor = torch.zeros(self.max_num_people, 4)
        people_bbox_norm_tensor = torch.zeros(self.max_num_people, 4)

        # crop people images
        vid_id, seq_id, img_file_name = img_file_path.split('/')[2:]
        img_id = img_file_name.split('.')[0]
        people_bbox_str = self.person_bbox_dic[f'{vid_id}_{seq_id}'][img_id]
        people_bbox = np.array(list(people_bbox_str.values()), dtype=np.int)
        people_bbox_norm = np.array(list(people_bbox_str.values()), dtype=np.float)
        for person_idx in range(people_bbox.shape[0]):
            img_person = img.crop(people_bbox[person_idx, :])
            if self.transforms_person:
                img_person = self.transforms_person(img_person)
            img_person_tensor[person_idx, :, :, :] = img_person
            att_inside_flag[person_idx] = 1

        # transform tensor
        if self.transforms_rgb:
            rgb_tensor = self.transforms_rgb(img)
        if self.transforms_rgb_wo_norm:
            rgb_tensor_wo_norm = self.transforms_rgb_wo_norm(img)
        if self.transforms_gt:
            img_gt = torch.tensor(img_gt).float()
            img_gt = self.transforms_gt(img_gt)
            img_gt = img_gt.expand(self.max_num_people, self.resize_height, self.resize_width)

        # generate gt box id for joint attention estimation
        gt_box_id_original = torch.tensor(self.gt_bbox_id[idx])
        gt_box_id_original_num = gt_box_id_original.shape[0]
        gt_box_id_original_max = 0 if gt_box_id_original_num == 0 else torch.max(gt_box_id_original)

        gt_box_id_expand_num = self.max_num_people - gt_box_id_original_num
        gt_box_id_expand = torch.tensor([(gt_box_id_original_max+i+1) for i in range(gt_box_id_expand_num)]).view(-1, 1)
        gt_box_id = torch.cat([gt_box_id_original, gt_box_id_expand], dim=0).long()

        people_bbox_img_num = people_bbox.shape[0]
        people_bbox_tensor[:people_bbox_img_num, :] = torch.tensor(people_bbox)
        people_bbox_norm_tensor[:people_bbox_img_num, :] = torch.tensor(people_bbox_norm)
        people_bbox_norm_tensor[:, ::2] /= img_width
        people_bbox_norm_tensor[:, 1::2] /= img_height

        # pack one data into a dict
        data = {}
        
        data['head_img'] = head_img
        data['head_feature'] = head_feature_tensor
        data['head_bbox'] = head_bbox_tensor
        data['head_vector_gt'] = head_vector_gt_tensor
        data['img_gt'] = img_gt
        data['gt_box'] = gt_box_expand
        data['gt_box_id'] = gt_box_id
        data['rgb_img'] = rgb_tensor
        data['saliency_img'] = rgb_tensor
        data['att_inside_flag'] = att_inside_flag
        data['rgb_path'] = img_file_path
        data['rgb_img_person'] = img_person_tensor
        data['people_bbox'] = people_bbox_tensor
        data['people_bbox_norm'] = people_bbox_norm_tensor

        if self.model_type == 'ball_detection' or self.model_type == 'isa' and self.data_name == 'volleyball':
            data['rgb_img_wo_norm'] = rgb_tensor_wo_norm

        return data

    # generage gt imgs for probability heatmap
    def load_gt_imgs_ball(self, img_width, img_height, bbox, gamma):

        x_min, y_min, x_max, y_max = map(int, bbox)
        x_center = int((x_max + x_min)//2)
        y_center = int((y_max + y_min)//2)

        # no ball image
        if x_min < 0:
            return np.zeros((self.max_num_people, img_height, img_width))

        gt_gaussian = self.generate_2d_gaussian(img_height, img_width, (x_center, y_center), gamma)

        return gt_gaussian

    # get a ball location
    def read_ball_bbox_from_csv(self, csv_file_path: str) -> list:
        try:
            df_anno = pd.read_csv(csv_file_path,  header=None)
        except pd.errors.EmptyDataError:
            return np.zeros((0, 4)), np.zeros((0, 2))
        
        bbox_array = np.zeros((df_anno.shape[0], 4))
        bbox_flag_array = np.zeros((df_anno.shape[0], 2))

        for img_idx in range(df_anno.shape[0]):
            anno_row = df_anno.iloc[img_idx, :].values[0].split(" ")
            x_min_ball, y_min_ball, x_max_ball, y_max_ball = map(int, anno_row[1:5])
            lost, occluded = map(int, anno_row[6:8])
            bbox_array[img_idx, :] = [x_min_ball, y_min_ball, x_max_ball, y_max_ball]

        return bbox_array, bbox_flag_array

    # get a ball location
    def get_ball_bbox_from_csv(self, bbox_array: list, img_idx: int) -> list:
        bbox = bbox_array[img_idx, :]
        
        return bbox

    def get_person_bbox_from_txt(self, txt_file_path: str) -> list:
        person_bbox_dic = {}

        with open(txt_file_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            per_id, x_min, y_min, x_max, y_max, img_id, _, _, _, iar_label = line.split()

            if not img_id in person_bbox_dic.keys():
                person_bbox_dic[img_id] = {}
            person_bbox_dic[img_id][per_id] = [x_min, y_min, x_max, y_max]

        return person_bbox_dic