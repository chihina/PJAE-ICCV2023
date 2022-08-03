import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F

class HeadPoseEstimatorResnet(nn.Module):
    def __init__(self, cfg):
        super(HeadPoseEstimatorResnet, self).__init__()

        # load vgg models
        vgg16 = models.vgg16(pretrained=True)
        vgg16.classifier = vgg16.classifier[:-1]
        self.vgg16 = vgg16

        # load head pose estimator
        self.head_pose_estimator = nn.Sequential(
            nn.Linear(4096, 2),
        )
        self.head_pose_tanh = nn.Tanh()

        # define loss function
        self.use_gaze_loss = cfg.exp_params.use_gaze_loss
        self.loss_func_head_pose = nn.MSELoss(reduction='sum')

    def forward(self, inp):
        # unpack input data
        head_img = inp['head_img']

        # backbone
        batch_size, people_num, channel_num, img_height, img_width = head_img.shape
        head_img = head_img.view(-1, channel_num, img_height, img_width)
        head_img = self.vgg16(head_img)

        # head pose estimation
        head_vector = self.head_pose_estimator(head_img)
        head_vector = self.head_pose_tanh(head_vector)
        head_vector = head_vector.view(batch_size, people_num, -1)
        head_img = head_img.view(batch_size, people_num, -1)

        # normarize head pose
        head_vector = F.normalize(head_vector, dim=-1)

        # pack output data
        out = {}
        out['head_vector'] = head_vector
        out['head_enc_map'] = head_img

        return out

    def calc_loss(self, inp, out):
        # unpack data
        head_vector_gt = inp['head_vector_gt']
        att_inside_flag = inp['att_inside_flag']
        head_vector = out['head_vector']

        # define coeficient
        if self.use_gaze_loss:
            loss_head_coef = 0.01
        else:
            loss_head_coef = 0

        # calculate loss
        head_vector_no_pad = head_vector[:, :, 0:2]*att_inside_flag[:, :, None]
        head_vector_gt_no_pad = head_vector_gt[:, :, 0:2]*att_inside_flag[:, :, None]
        head_num_sum_no_pad = torch.sum(att_inside_flag)
        loss_head = self.loss_func_head_pose(head_vector_no_pad, head_vector_gt_no_pad)
        loss_head = loss_head/head_num_sum_no_pad
        loss_head = loss_head*loss_head_coef

        # pack data
        loss_set = {}
        loss_set['loss_head'] = loss_head

        return loss_set