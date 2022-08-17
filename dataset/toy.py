'''
    Author: Chihiro Nakatani
    Dec 7th, 2021
'''

# import dl lib
import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms, models

# import general lib
import numpy as np
import os
import sys
import glob
import cv2
import scipy.io as sio
import math
from PIL import Image
import matplotlib
matplotlib.use('Agg')

class ToyDataset(Dataset):
    def __init__(self, cfg, mode, transform=None, pre_transform=None):

        # dataset variables
        self.mode = mode
        self.dataset_dir = cfg.data.dataset_dir

        # exp settings
        self.wandb_name = cfg.exp_set.wandb_name
        self.resize_width = cfg.exp_set.resize_width
        self.resize_height = cfg.exp_set.resize_height
        self.resize_head_width = cfg.exp_set.resize_head_width
        self.resize_head_height = cfg.exp_set.resize_head_height
        self.pad_rate = 0

        # exp params
        self.gaussian_sigma = cfg.exp_params.gaussian_sigma
        self.use_frame_type = cfg.exp_params.use_frame_type

        # experiment parameters
        self.feature_list = []
        self.head_bbox_list = []
        self.gt_bbox = []
        self.rgb_path_list = []
        self.saliency_path_list = []
        self.att_inside_list = []

        # make dataset
        self.generate_dataset_list()

        # set maximum number of people
        self.max_num_people = 0
        self.set_max_num_people()

        self.transforms = transforms.Compose(
            [
                transforms.Resize((self.resize_head_height, self.resize_head_width)),
                transforms.ToTensor(),
                # transforms.Normalize(
                #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                # ),
            ]
        )

        self.transforms_rgb = transforms.Compose(
            [
                transforms.Resize((self.resize_height, self.resize_width)),
                transforms.ToTensor(),
                # transforms.Normalize(
                #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                # ),
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

        # get path of images_nk
        img_file_path = self.rgb_path_list[idx]
        saliency_file_path = self.saliency_path_list[idx]

        # read a rgb image
        img = Image.open(img_file_path)
        img_width, img_height = img.size

        # generate gt heatmaps
        bboxes_origin = np.copy(self.gt_bbox[idx])
        gt_img = self.load_gt_imgs_ball(img_width, img_height, bboxes_origin, self.gaussian_sigma)
        gt_img = torch.tensor(gt_img).float()

        # normalize gt bbox
        bboxes = np.zeros((self.max_num_people, 4))
        bboxes[:bboxes_origin.shape[0], :] = np.copy(self.gt_bbox[idx])
        bboxes[:, ::2] /= img_width
        bboxes[:, 1::2] /= img_height

        # initialize tensors
        head_img = torch.zeros(self.max_num_people, 3, self.resize_head_height, self.resize_head_width)
        head_vector_gt_tensor = torch.zeros(self.max_num_people, 2)
        head_feature_tensor = torch.zeros(self.max_num_people, 2)
        head_bbox_tensor = torch.zeros(self.max_num_people, 4)

        # crop heads in images_nk
        for head_idx in range(len(self.feature_list[idx])):
            head_x, head_y = map(int, self.feature_list[idx][head_idx])
            head_x_min, head_y_min, head_x_max, head_y_max = map(int, self.head_bbox_list[idx][head_idx])
            x_mid = (bboxes_origin[head_idx, 0]+bboxes_origin[head_idx, 2])/2
            y_mid = (bboxes_origin[head_idx, 1]+bboxes_origin[head_idx, 3])/2
            head_ball_vec_x, head_ball_vec_y = x_mid-head_x, y_mid-head_y
            head_ball_vec_norm = ((head_ball_vec_x**2+head_ball_vec_y**2) ** 0.5) + 1e-5
            head_ball_vec_x, head_ball_vec_y = head_ball_vec_x/head_ball_vec_norm, head_ball_vec_y/head_ball_vec_norm
            croped_head = img.crop((head_x_min, head_y_min, head_x_max, head_y_max))

            # transform tensor 
            if self.transforms:
                croped_head = self.transforms(croped_head)

            head_img[head_idx, :, :, :] = croped_head
            head_feature_tensor[head_idx, :2] = torch.tensor(self.feature_list[idx][head_idx])
            head_vector_gt_tensor[head_idx, :2] = torch.tensor([head_ball_vec_x, head_ball_vec_y])
            head_bbox_tensor[head_idx, :4] = torch.tensor(list(map(int, self.head_bbox_list[idx][head_idx])))

        head_feature_tensor[:, 0] /= img_width
        head_feature_tensor[:, 1] /= img_height

        # transform tensor 
        if self.transforms_rgb:
            rgb_tensor = self.transforms_rgb(Image.open(img_file_path))
        if self.transforms_gt:
            gt_img = self.transforms_gt(gt_img)
        if self.transforms_saliency:
            saliency_tensor = self.transforms_saliency(Image.open(saliency_file_path))

        # generate gt box id for joint attention estimation
        gt_box_id_original = torch.tensor([[0]])
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
        data['head_bbox_tensor'] = head_bbox_tensor
        data['att_inside_flag'] = torch.sum(torch.tensor(bboxes), dim=-1)!=0
        data['rgb_path'] = img_file_path

        return data

    # set maximum number of people
    def set_max_num_people(self):
        for data_idx in range(len(self.feature_list)):
            data_people_num = len(self.feature_list[data_idx])
            self.max_num_people = max(self.max_num_people, data_people_num)

    # read graph information
    def generate_dataset_list(self):
        img_idx = 0
        for img_name in sorted(os.listdir(os.path.join(self.dataset_dir, 'images', self.mode))):
            img_idx += 1

            if self.use_frame_type == 'mid' and img_idx % 10 != 0:
                continue
            else:
                pass

            data_id = os.path.splitext(img_name)[0]            
            img_path = os.path.join(self.dataset_dir, 'images', self.mode, f'{data_id}.jpg')
            ann_path_gt = os.path.join(self.dataset_dir, 'annotations', self.mode, f'{data_id}_gt.txt')
            ann_path_head = os.path.join(self.dataset_dir, 'annotations', self.mode, f'{data_id}_head.txt')

            self.rgb_path_list.append(img_path)
            self.saliency_path_list.append(img_path)

            with open(ann_path_gt, 'r') as f:
                gt_bbox = list(map(int, f.readlines()[0].split()))
                gt_bbox = np.array(gt_bbox).reshape(-1, 4)
                

            with open(ann_path_head, 'r') as f:
                head_bbox = list(map(int, f.readlines()[0].split()))
                head_bbox = np.array(head_bbox).reshape(-1, 4)

            self.gt_bbox.append(gt_bbox)
            self.head_bbox_list.append(head_bbox)

            # calculate each person head position
            use_head_feature = np.zeros((head_bbox.shape[0], 2))
            if head_bbox.shape[0] != 0:
                use_head_feature[:, 0] = (head_bbox[:, 0] + head_bbox[:, 2]) / 2
                use_head_feature[:, 1] = (head_bbox[:, 1] + head_bbox[:, 3]) / 2
            self.feature_list.append(use_head_feature)

    # generage gt imgs for probability heatmap
    def load_gt_imgs_ball(self, img_width, img_height, bbox, gamma):
        gt_gaussian = np.zeros((1, img_height, img_width), dtype=np.float32)
        for co_idx in range(len(bbox)):
            x_min, y_min, x_max, y_max = map(int, bbox[co_idx])

            # if positive sample
            if np.sum(bbox[co_idx]) != 0:
                x_center, y_center = int((x_max + x_min)//2), int((y_max + y_min)//2)
                gt_gaussian += self.generate_2d_gaussian(img_height, img_width, (x_center, y_center), gamma)

        return gt_gaussian

    # generator gt gaussian
    def generate_2d_gaussian(self, height, width, peak_xy, sigma=1):
        peak_x, peak_y = peak_xy
        x = np.arange(0, width, 1)
        y = np.arange(0, height, 1)
        X,Y = np.meshgrid(x,y)
        heatmap = np.exp(- ((X-peak_x) ** 2 + (Y-peak_y) ** 2) / (2 * sigma ** 2))

        return heatmap.reshape(1, heatmap.shape[0], heatmap.shape[1])