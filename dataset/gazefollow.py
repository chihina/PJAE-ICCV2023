import torch
from torch.utils.data import Dataset
from torchvision import transforms

import numpy as np
import os
import sys
from PIL import Image
import matplotlib
from tqdm import tqdm
matplotlib.use('Agg')

class GazeFollowDataset(Dataset):
    def __init__(self, cfg, mode, transform=None, pre_transform=None):

        # dataset variables
        self.dataset_dir = cfg.data.dataset_dir
        self.mode = mode

        # data parameters
        self.wandb_name = cfg.exp_set.wandb_name
        self.resize_width = cfg.exp_set.resize_width
        self.resize_height = cfg.exp_set.resize_height
        self.resize_head_width = cfg.exp_set.resize_head_width
        self.resize_head_height = cfg.exp_set.resize_head_height

        # gt parameters
        self.gaussian_sigma = cfg.exp_params.gaussian_sigma

        # experiment parameters
        self.feature_list = []
        self.head_bbox_list = []
        self.gt_bbox = []
        self.gt_bbox_id = []

        self.rgb_path_list = []
        self.saliency_path_list = []
        self.att_inside_list = []

        # make dataset
        self.generate_dataset_list()

        # set maximum number of people
        self.max_num_people = 0
        self.set_max_num_people()

        self.transforms_head = transforms.Compose(
            [
                transforms.Resize((self.resize_head_height, self.resize_head_width)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.transforms_rgb = transforms.Compose(
            [
                transforms.Resize((self.resize_height, self.resize_width)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.transforms_gt = transforms.Compose(
            [
                transforms.Resize((self.resize_height, self.resize_width)),
            ]
        )

        self.transforms_saliency = transforms.Compose(
            [
                transforms.Resize((self.resize_height, self.resize_width)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.rgb_path_list)

    def __getitem__(self, idx):

        # get image path
        img_file_path = self.rgb_path_list[idx]
        saliency_file_path = self.saliency_path_list[idx]

        # open images
        img = Image.open(img_file_path).convert('RGB')
        img_width, img_height = img.size

        # generate gt heatmaps
        bboxes_origin = np.copy(self.gt_bbox[idx])
        bboxes_origin[:, ::2] *= img_width
        bboxes_origin[:, 1::2] *= img_height
        gt_img = self.load_gt_img(img_width, img_height, bboxes_origin, self.gaussian_sigma)
        gt_img = torch.tensor(gt_img).float()
        att_inside_flag = np.zeros((self.max_num_people))
        att_inside_flag[:bboxes_origin.shape[0]] = np.copy(self.att_inside_list[idx])[:, 0]

        # normalize gt bbox
        bboxes = np.zeros((self.max_num_people, 4))
        bboxes[:bboxes_origin.shape[0], :] = np.copy(self.gt_bbox[idx])

        # initialize tensors
        head_img = torch.zeros(self.max_num_people, 3, self.resize_head_height, self.resize_head_width)
        head_feature_tensor = torch.zeros(self.max_num_people, 2)
        head_bbox_tensor = torch.zeros(self.max_num_people, 4)
        head_vector_gt_tensor = torch.zeros(self.max_num_people, 2)

        # get features of heads
        head_num = self.feature_list[idx].shape[0]
        for head_idx in range(head_num):
            x_mid = (bboxes_origin[head_idx, 0]+bboxes_origin[head_idx, 2])/2
            y_mid = (bboxes_origin[head_idx, 1]+bboxes_origin[head_idx, 3])/2
            head_x, head_y = map(float, self.feature_list[idx][head_idx])
            head_x, head_y = head_x*img_width, head_y*img_height
            head_ball_vec_x, head_ball_vec_y = x_mid-head_x, y_mid-head_y
            head_ball_vec_norm = ((head_ball_vec_x**2+head_ball_vec_y**2) ** 0.5) + 1e-5
            head_ball_vec_x, head_ball_vec_y = head_ball_vec_x/head_ball_vec_norm, head_ball_vec_y/head_ball_vec_norm
            head_x_min, head_y_min, head_x_max, head_y_max = map(float, self.head_bbox_list[idx][head_idx])
            head_x_min, head_x_max = map(lambda x: int(x*img_width), [head_x_min, head_x_max])
            head_y_min, head_y_max = map(lambda x: int(x*img_height), [head_y_min, head_y_max])
            croped_head = img.crop((head_x_min, head_y_min, head_x_max, head_y_max))

            # transform tensor
            if self.transforms_head:
                croped_head = self.transforms_head(croped_head)
            head_img[head_idx, :, :, :] = croped_head

            head_feature_tensor[head_idx, :2] = torch.tensor(self.feature_list[idx][head_idx])
            head_vector_gt_tensor[head_idx, :2] = torch.tensor([head_ball_vec_x, head_ball_vec_y])
            head_bbox_tensor[head_idx, :4] = torch.tensor([head_x_min, head_y_min, head_x_max, head_y_max])

        # normalize head position
        head_bbox_tensor[:, ::2] /= img_width
        head_bbox_tensor[:, 1::2] /= img_height

        # transform tensor 
        if self.transforms_rgb:
            rgb_tensor = self.transforms_rgb(img)
        if self.transforms_gt:
            gt_img = self.transforms_gt(gt_img)
        if self.transforms_saliency:
            saliency_tensor = self.transforms_saliency(img)

        # generate gt box id for joint attention estimation
        gt_box_id_original = torch.tensor(self.gt_bbox_id[idx])
        gt_box_id_original_num = gt_box_id_original.shape[0]
        gt_box_id_original_max = torch.max(gt_box_id_original)
        gt_box_id_expand_num = self.max_num_people - gt_box_id_original_num
        gt_box_id_expand = torch.tensor([(gt_box_id_original_max+i+1) for i in range(gt_box_id_expand_num)]).view(-1, 1)
        gt_box_id = torch.cat([gt_box_id_original, gt_box_id_expand], dim=0).long()

        data = {}
        data['head_img'] = head_img
        data['head_feature'] = head_feature_tensor
        data['head_vector_gt'] = head_vector_gt_tensor
        data['img_gt'] = gt_img
        data['gt_box'] = bboxes
        data['gt_box_id'] = gt_box_id
        data['rgb_img'] = rgb_tensor
        data['saliency_img'] = saliency_tensor
        data['head_bbox'] = head_bbox_tensor
        data['att_inside_flag'] = att_inside_flag
        data['rgb_path'] = img_file_path

        return data

    # set maximum number of people
    def set_max_num_people(self):
        for data_idx in range(len(self.feature_list)):
            data_people_num = len(self.feature_list[data_idx])
            self.max_num_people = max(self.max_num_people, data_people_num)

    # read graph information
    def generate_dataset_list(self):
        ann_file_path = os.path.join(self.dataset_dir, 'data_new', f'{self.mode}_annotations.txt')
        with open(ann_file_path, 'r') as f:
            lines = f.readlines()

        img_id_list = []
        line_idx = 0
        for line in tqdm(lines):
            line_idx += 1
            # if line_idx % 10 != 0:
                # continue

            line_comp = line.split(',')
            img_name, img_index, head_x_min, head_y_min, head_width, head_height, eye_x, eye_y, gaze_x, gaze_y, _, _ = line_comp

            if self.mode == 'train':
                img_id = img_index
            else:
                img_id, _ = img_index.split('-')
            if not img_id in img_id_list:
                img_id_list.append(img_id)
            else:
                continue

            body_x_min, body_y_min, body_width, body_height, eye_x, eye_y, gaze_x, gaze_y = map(float, [head_x_min, head_y_min, head_width, head_height, eye_x, eye_y, gaze_x, gaze_y])
            img_path = os.path.join(self.dataset_dir, 'data_new', img_name)

            head_width, head_height = body_width*0.2, body_height*0.15
            head_x_min, head_x_max = eye_x-head_width, eye_x+head_width
            head_y_min, head_y_max = eye_y-head_height, eye_y+head_height
            # head_x_max, head_y_max = head_x_min+head_width, head_y_min+head_height
            head_bbox = [head_x_min, head_y_min, head_x_max, head_y_max]
            att_point = [gaze_x, gaze_y]
            att_inside = True

            # get head features
            head_bbox = np.array(head_bbox).reshape(-1, 4)
            use_head_feature = np.zeros((head_bbox.shape[0], 2))
            if head_bbox.shape[0] != 0:
                # use_head_feature[:, 0] = (head_bbox[:, 0] + head_bbox[:, 2]) / 2
                # use_head_feature[:, 1] = (head_bbox[:, 1] + head_bbox[:, 3]) / 2
                use_head_feature[:, 0] = eye_x
                use_head_feature[:, 1] = eye_y

            # get attention points
            gt_point = np.array(att_point).reshape(-1, 2)
            gt_bbox = np.zeros((gt_point.shape[0], 4))
            box_size = 0.1
            gt_bbox[:, 0:2], gt_bbox[:, 2:4] = gt_point - box_size, gt_point + box_size
            att_inside = np.array([att_inside]).reshape(-1, 1)
            gt_bbox_idx = np.arange(gt_bbox.shape[0]).reshape(-1, 1)

            # append information of one image
            self.rgb_path_list.append(img_path)
            self.saliency_path_list.append(img_path)
            self.head_bbox_list.append(head_bbox)
            self.feature_list.append(use_head_feature)
            self.gt_bbox.append(gt_bbox)
            self.gt_bbox_id.append(gt_bbox_idx)
            self.att_inside_list.append(att_inside)

            if self.wandb_name == 'debug':
                if line_idx > 100:
                    break

    # generage gt imgs for probability heatmap
    def load_gt_img(self, img_width, img_height, bbox, gamma):
        gt_gaussian = np.zeros((self.max_num_people, img_height, img_width))
        gt_box_num = bbox.shape[0]
        # generate a heatmap of each attention
        for co_idx in range(gt_box_num):
            x_min, y_min, x_max, y_max = map(int, bbox[co_idx])
            x_center, y_center = int((x_max + x_min)//2), int((y_max + y_min)//2)
            gt_gaussian_one = self.generate_2d_gaussian(img_height, img_width, (x_center, y_center), gamma)
            gt_gaussian[co_idx, :, :] =  gt_gaussian_one

        return gt_gaussian

    # generator gt gaussian
    def generate_2d_gaussian(self, height, width, peak_xy, sigma=1):
        peak_x, peak_y = peak_xy
        x = np.arange(0, width, 1)
        y = np.arange(0, height, 1)
        X,Y = np.meshgrid(x,y)
        heatmap = np.exp(- ((X-peak_x) ** 2 + (Y-peak_y) ** 2) / (2 * sigma ** 2))

        return heatmap.reshape(1, heatmap.shape[0], heatmap.shape[1])