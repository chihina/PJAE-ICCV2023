import torch
import torch.nn as nn
import torchvision.models as models
import sys

from .ConvLSTM_pytorch.convlstm import ConvLSTM

class InferringSharedAttentionEstimator(nn.Module):
    def __init__(self, cfg):
        super(InferringSharedAttentionEstimator, self).__init__()

        ## set useful variables
        self.epsilon = 1e-7
        self.pi = 3.1415

        ## set data
        self.dataset_name = cfg.data.name

        ## exp settings
        self.resize_width = cfg.exp_set.resize_width
        self.resize_height = cfg.exp_set.resize_height

        self.gpu_list = range(cfg.exp_set.gpu_start, cfg.exp_set.gpu_finish+1)
        self.device = torch.device(f"cuda:{self.gpu_list[0]}")
        self.wandb_name = cfg.exp_set.wandb_name
        self.batch_size = cfg.exp_set.batch_size

        # exp params
        self.use_frame_type = cfg.exp_params.use_frame_type
        self.use_frame_num = 1
        if self.use_frame_type == 'all':
            if 'volley' in self.dataset_name:
                self.use_frame_num = 10
            else:
                assert False, 'Not implemented use frame num'

        # define loss function
        self.loss = cfg.exp_params.loss
        if self.loss == 'mse':
            print('Use MSE loss function')
            self.loss_func_joint_attention = nn.MSELoss()
        elif self.loss == 'bce':
            print('Use BCE loss function')
            self.loss_func_joint_attention = nn.BCELoss()
        elif self.loss == 'l1':
            print('Use l1 loss function')
            self.loss_func_joint_attention = nn.L1Loss()
        self.use_e_att_loss = cfg.exp_params.use_e_att_loss

        self.conv_in_channels = 1+1
        self.spatial_detection_module = nn.Sequential(
            nn.Conv2d(in_channels=self.conv_in_channels, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=1),
            # nn.Sigmoid(),
        )

        if self.use_frame_type == 'all':
            conv_lstm = ConvLSTM(input_dim=1,
                            hidden_dim=[40, 40, 40, 40, 1],
                            kernel_size=[(3, 3), (3, 3), (3, 3), (3, 3), (1, 1)],
                            num_layers=5,
                            batch_first=True,
                            bias=True,
                            return_all_layers=True
                            )

            self.temporal_detection_module = nn.Sequential(
                conv_lstm,
            )

    def forward(self, inp):
        head_vector = inp['head_vector']
        head_feature = inp['head_feature']
        xy_axis_map = inp['xy_axis_map']
        head_xy_map = inp['head_xy_map']
        gaze_xy_map = inp['gaze_xy_map']
        saliency_img = inp['saliency_img']

        self.batch_size, self.frame_num, self.people_num, _ = head_feature.shape
        _, _, _, self.hm_height, self.hm_width = saliency_img.shape
        people_exist_mask = (torch.sum(head_feature, dim=-1) != 0).bool()
        people_exist_num = torch.sum(people_exist_mask, dim=-1)

        # generate head xy map
        head_xy_map = head_xy_map.unsqueeze(1)
        head_xy_map = head_xy_map.expand(self.batch_size, self.frame_num, self.people_num, 2, self.hm_height, self.hm_width)
        head_xy_map = head_xy_map * head_feature[:, :, :, :2, None, None]

        # generate gaze xy map
        gaze_xy_map = gaze_xy_map.unsqueeze(1)
        gaze_xy_map = gaze_xy_map.expand(self.batch_size, self.frame_num, self.people_num, 2, self.hm_height, self.hm_width)
        gaze_xy_map = gaze_xy_map * head_vector[:, :, :, :2, None, None]

        # expand xy axis map
        xy_axis_map = xy_axis_map.unsqueeze(1)
        xy_axis_map = xy_axis_map.expand(self.batch_size, self.frame_num, self.people_num, 2, self.hm_height, self.hm_width)

        # generate gaze cone map
        xy_axis_map_dif_head = xy_axis_map - head_xy_map
        x_axis_map_dif_head_mul_gaze = xy_axis_map_dif_head * gaze_xy_map
        xy_dot_product = torch.sum(x_axis_map_dif_head_mul_gaze, dim=-3)
        xy_dot_product = xy_dot_product / (torch.norm(xy_axis_map_dif_head, dim=-3) + self.epsilon)
        xy_dot_product = xy_dot_product / (torch.norm(gaze_xy_map, dim=-3) + self.epsilon)

        # calculate theta and distance map
        theta_x_y = torch.acos(torch.clamp(xy_dot_product, -1+self.epsilon, 1-self.epsilon))

        # generate sigma of gaussian
        # multiply zero to padding maps
        self.gaussian_sigma = 0.5
        angle_dist = torch.exp(-torch.pow(theta_x_y, 2)/(2*self.gaussian_sigma**2)) / self.gaussian_sigma
        angle_dist = angle_dist * (torch.sum(head_feature, dim=-1) != 0)[:, :, :, None, None]

        # sum all gaze maps (divide people num excluding padding people)
        angle_dist_sum_pooling = torch.sum(angle_dist, dim=2)[:, :, None, :, :]
        angle_dist_sum_pooling = angle_dist_sum_pooling/people_exist_num[:, :, None, None, None]

        # cat angle img and saliency img
        # spatial detection module
        angle_saliency_img = torch.cat([angle_dist_sum_pooling, saliency_img], dim=-3)
        angle_saliency_img = angle_saliency_img.reshape(self.batch_size*self.frame_num, 1+1, self.hm_height, self.hm_width)
        estimated_joint_attention = self.spatial_detection_module(angle_saliency_img)
        estimated_joint_attention = estimated_joint_attention.reshape(self.batch_size, self.frame_num, 1, self.hm_height, self.hm_width)

        if self.use_frame_type == 'all':
            estimated_joint_attention = estimated_joint_attention.reshape(self.batch_size, self.frame_num, 1, self.hm_height, self.hm_width)
            layer_output, last_states = self.temporal_detection_module(estimated_joint_attention)
            estimated_joint_attention = layer_output[-1].reshape(self.batch_size, self.frame_num, 1, self.hm_height, self.hm_width)

        # return final img
        estimated_joint_attention = estimated_joint_attention[:, :, 0, :, :]

        # pack return values
        data = {}
        data['head_tensor'] = head_vector
        data['img_pred'] = estimated_joint_attention
        data['angle_dist'] = angle_dist
        data['angle_dist_pool'] = angle_dist_sum_pooling
        data['saliency_map'] = saliency_img

        data['person_person_attention_heatmap'] = estimated_joint_attention.unsqueeze(2).expand(-1, -1, self.people_num, -1, -1)
        data['person_person_joint_attention_heatmap'] = estimated_joint_attention
        data['person_scene_attention_heatmap'] = estimated_joint_attention.unsqueeze(2).expand(-1, -1, self.people_num, -1, -1)
        data['person_scene_joint_attention_heatmap'] = estimated_joint_attention
        data['final_joint_attention_heatmap'] = estimated_joint_attention

        return data

    def calc_loss(self, inp, out, cfg):
        # unpack data
        img_gt = inp['img_gt']
        att_inside_flag = inp['att_inside_flag']
        img_pred = out['img_pred']

        # switch loss coeficient
        if self.use_e_att_loss:
            loss_map_coef = 1
        else:
            loss_map_coef = 0

        # calculate final map loss
        img_gt_att = torch.sum(img_gt, dim=-3)
        img_gt_att_thresh = torch.ones(1, dtype=img_gt_att.dtype, device=img_gt_att.device)        
        img_gt_att = torch.where(img_gt_att>img_gt_att_thresh, img_gt_att_thresh, img_gt_att)

        loss_map = self.loss_func_joint_attention(img_pred.float(), img_gt_att.float())
        loss_map = loss_map_coef * loss_map

        loss_set = {}
        loss_set['loss_map'] = loss_map

        return loss_set