import torch
import torch.nn as nn
from torch.nn import functional as F
import sys
import  numpy as np

class JointAttentionFusion(nn.Module):
    def __init__(self, cfg):
        super(JointAttentionFusion, self).__init__()

        self.gpu_list = range(cfg.exp_set.gpu_start, cfg.exp_set.gpu_finish+1)
        self.device = torch.device(f"cuda:{self.gpu_list[0]}")
        self.wandb_name = cfg.exp_set.wandb_name
        self.batch_size = cfg.exp_set.batch_size
        self.resize_width = cfg.exp_set.resize_width
        self.resize_height = cfg.exp_set.resize_height

        # define person-to-scene relation extractor
        self.p_s_estimator_type = cfg.model_params.p_s_estimator_type
        if self.p_s_estimator_type == 'cnn' or self.p_s_estimator_type == 'transformer':
            self.rgb_cnn_extractor_stage_idx = cfg.model_params.rgb_cnn_extractor_stage_idx
            down_scale_list = [2, 4, 8, 16, 32]
            down_scale_ratio = down_scale_list[self.rgb_cnn_extractor_stage_idx]
            self.hm_height_p_s = self.resize_height
            self.hm_width_p_s = self.resize_width
            self.hm_height_middle_p_s = self.resize_height
            self.hm_width_middle_p_s = self.resize_width
        elif self.p_s_estimator_type == 'davt':
            self.hm_height_p_s = self.resize_height
            self.hm_width_p_s = self.resize_width
            down_scale_ratio = 8
            self.hm_height_middle_p_s = 64
            self.hm_width_middle_p_s = 64

        # define loss function
        self.loss = cfg.exp_params.loss
        if self.loss == 'mse':
            print('Use MSE loss function')
            self.loss_func_hm_mean = nn.MSELoss(reduction='mean')
            self.loss_func_hm_sum = nn.MSELoss(reduction='sum')
        elif self.loss == 'bce':
            print('Use BCE loss function')
            self.loss_func_hm_mean = nn.BCELoss(reduction='mean')
            self.loss_func_hm_sum = nn.BCELoss(reduction='sum')
        
        self.fusion_net_type = cfg.model_params.fusion_net_type
        self.use_final_jo_att_loss = cfg.exp_params.use_final_jo_att_loss

        if self.loss == 'mse':
            final_activation_layer = nn.Identity()
        elif self.loss == 'bce':
            final_activation_layer = nn.Sigmoid()

        if self.fusion_net_type == 'early':
            self.person_person_preconv = nn.Sequential(
                nn.Identity(),
            )
            self.person_scene_preconv = nn.Sequential(
                nn.Identity(),
            )
            self.final_joint_atention_heatmap = nn.Sequential(
                nn.Conv2d(in_channels=2, out_channels=8, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.ConvTranspose2d(8, 8, 4, 2, 1, bias=False),
                nn.ReLU(),
                nn.ConvTranspose2d(8, 8, 4, 2, 1, bias=False),
                nn.ReLU(),
                nn.Conv2d(in_channels=8, out_channels=1, kernel_size=1),
                final_activation_layer,
            )
        elif self.fusion_net_type == 'mid':
            self.person_person_preconv = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
                nn.Conv2d(in_channels=4, out_channels=8, kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
                nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
            )
            self.person_scene_preconv = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
                nn.Conv2d(in_channels=4, out_channels=8, kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
                nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
            )
            self.final_joint_atention_heatmap = nn.Sequential(
                nn.Conv2d(in_channels=16, out_channels=8, kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
                nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
                nn.Conv2d(in_channels=8, out_channels=1, kernel_size=1),
                final_activation_layer,
            )
        elif self.fusion_net_type == 'late':
            pass
        elif self.fusion_net_type == 'simple_average':
            self.final_fusion_weight = nn.Parameter(torch.rand(2))
        elif self.fusion_net_type == 'scalar_weight':
            self.final_fusion_weight = nn.Parameter(torch.rand(2))
            self.final_fusion_softmax = nn.Softmax()
        else:
            print('please use correct fusion net type')
            self.person_person_preconv = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            )
            self.person_scene_preconv = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            )
            self.final_joint_atention_heatmap = nn.Sequential(
                nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=8, out_channels=1, kernel_size=1),
                final_activation_layer,
            )

    def forward(self, inp):

        head_vector = inp['head_vector']
        head_feature = inp['head_feature']
        xy_axis_map = inp['xy_axis_map']
        head_xy_map = inp['head_xy_map']
        gaze_xy_map = inp['gaze_xy_map']

        # get usuful variable
        self.batch_size, people_num, _, _, _ = xy_axis_map.shape

        # attention estimation of person-to-person path
        person_person_attention_heatmap = inp['person_person_attention_heatmap']

        # joint attention estimation of person-to-person path
        person_person_joint_attention_heatmap = inp['person_person_joint_attention_heatmap']
        
        # attention estimation of person-to-scene path
        person_scene_attention_heatmap = inp['person_scene_attention_heatmap']
        
        # joint attention estimation of person-to-scene path
        person_scene_joint_attention_heatmap = inp['person_scene_joint_attention_heatmap']

        # final joint attention estimation
        if self.fusion_net_type == 'simple_average':
            final_joint_attention_heatmap = (person_person_joint_attention_heatmap+person_scene_joint_attention_heatmap) / 2            
        elif self.fusion_net_type == 'scalar_weight':
            final_fusion_weight = self.final_fusion_weight
            final_fusion_weight = self.final_fusion_softmax(final_fusion_weight)
            final_fusion_weight_p_p = final_fusion_weight[0]
            final_fusion_weight_p_s = final_fusion_weight[1]
            final_joint_attention_heatmap = (final_fusion_weight_p_p*person_person_joint_attention_heatmap)+(final_fusion_weight_p_s*person_scene_joint_attention_heatmap)
            print(f'p-p:{final_fusion_weight_p_p.item():.2f}, p-s:{final_fusion_weight_p_s.item():.2f}')
        else:
            person_person_joint_attention_heatmap_preconv = self.person_person_preconv(person_person_joint_attention_heatmap)
            person_scene_joint_attention_heatmap_preconv = self.person_scene_preconv(person_scene_joint_attention_heatmap)
            dual_heatmap = torch.cat([person_person_joint_attention_heatmap_preconv, person_scene_joint_attention_heatmap_preconv], dim=1)
            final_joint_attention_heatmap = self.final_joint_atention_heatmap(dual_heatmap)

        # pack return values
        data = {}
        data['final_joint_attention_heatmap'] = final_joint_attention_heatmap

        return data

    def calc_loss(self, inp, out, cfg):
        # unpack data (input)
        img_gt_attention = inp['img_gt']
        gt_box = inp['gt_box']
        gt_box_id = inp['gt_box_id']
        att_inside_flag = inp['att_inside_flag']

        # unpack data (output)
        final_joint_attention_heatmap = out['final_joint_attention_heatmap']

        # switch loss coeficient
        self.use_final_jo_att_loss = cfg.exp_params.use_final_jo_att_loss
        self.final_jo_att_loss_weight = cfg.exp_params.final_jo_att_loss_weight
        if self.use_final_jo_att_loss:
            use_final_jo_att_coef = self.final_jo_att_loss_weight
        else:
            use_final_jo_att_coef = 0

        # generate gt map
        img_gt_attention = (img_gt_attention * att_inside_flag[:, :, None, None]).float()
        img_gt_joint_attention = torch.sum(img_gt_attention, dim=1)
        img_gt_all_thresh = torch.ones(1, device=img_gt_attention.device).float()
        img_gt_joint_attention = torch.where(img_gt_joint_attention>=img_gt_all_thresh, img_gt_all_thresh, img_gt_joint_attention)
        # print('F', f'{torch.min(final_joint_attention_heatmap).item():.2f} {torch.max(final_joint_attention_heatmap).item():.2f}')

        # calculate final loss
        final_joint_attention_heatmap = F.interpolate(final_joint_attention_heatmap, (self.resize_height, self.resize_width), mode='bilinear')
        final_joint_attention_heatmap = final_joint_attention_heatmap[:, 0, :, :]
        loss_final_jo_att = self.loss_func_hm_mean(final_joint_attention_heatmap.float(), img_gt_joint_attention.float())
        loss_final_jo_att = use_final_jo_att_coef * loss_final_jo_att
        # print('loss_final_jo_att', loss_final_jo_att)

        # pack loss
        loss_set = {}
        loss_set['loss_final_jo_att'] = loss_final_jo_att

        return loss_set

class JointAttentionFusionDummy(nn.Module):
    def __init__(self):
        super(JointAttentionFusionDummy, self).__init__()
        self.fc = nn.Linear(1, 1)

    def forward(self, inp):
        out = {}
        return out

    def calc_loss(self, inp, out, cfg):
        loss_set = {}
        return loss_set