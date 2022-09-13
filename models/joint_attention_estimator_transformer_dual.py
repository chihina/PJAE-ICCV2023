import torch
import torch.nn as nn
from torch.nn import functional as F
import sys
import  numpy as np

class JointAttentionEstimatorTransformerDual(nn.Module):
    def __init__(self, cfg):
        super(JointAttentionEstimatorTransformerDual, self).__init__()

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

        ## model params
        # position
        self.use_position = cfg.model_params.use_position

        # gaze
        self.use_gaze = cfg.model_params.use_gaze

        # action
        self.use_action = cfg.model_params.use_action

        # head embedding type
        self.head_embedding_type = cfg.model_params.head_embedding_type

        # whole image
        self.use_img = cfg.model_params.use_img

        # transformer
        self.people_feat_dim = cfg.model_params.people_feat_dim
        self.use_people_people_trans = cfg.model_params.use_people_people_trans
        self.people_people_trans_enc_num = cfg.model_params.people_people_trans_enc_num
        self.mha_num_heads_people_people = cfg.model_params.mha_num_heads_people_people

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
        
        self.use_person_person_att_loss = cfg.exp_params.use_person_person_att_loss
        self.use_person_person_jo_att_loss = cfg.exp_params.use_person_person_jo_att_loss
        self.use_person_scene_att_loss = cfg.exp_params.use_person_scene_att_loss
        self.use_person_scene_jo_att_loss = cfg.exp_params.use_person_scene_jo_att_loss
        self.use_final_jo_att_loss = cfg.exp_params.use_final_jo_att_loss

        embeding_param_num = 0
        if self.use_position:
            embeding_param_num += 2
        if self.use_action:
            embeding_param_num += 9
        if self.use_gaze:
            embeding_param_num += 2

        if self.head_embedding_type == 'liner':
            self.head_info_feat_embeding = nn.Sequential(
                nn.Linear(embeding_param_num, self.people_feat_dim),
            )
        elif self.head_embedding_type == 'mlp':
            self.head_info_feat_embeding = nn.Sequential(
                nn.Linear(embeding_param_num, self.people_feat_dim),
                nn.ReLU(),
                nn.Linear(self.people_feat_dim, self.people_feat_dim),
                nn.ReLU(),
                nn.Linear(self.people_feat_dim, self.people_feat_dim),
            )
        else:
            print('please use correct head embedding type')
            sys.exit()

        self.ja_embedding = nn.Parameter(torch.zeros(1, 1, self.people_feat_dim))
        if self.use_people_people_trans:
            print('Use tranformer encoder for people relation')
            self.people_people_self_attention = nn.ModuleList([nn.MultiheadAttention(embed_dim=self.people_feat_dim, num_heads=self.mha_num_heads_people_people, batch_first=True) for _ in range(self.people_people_trans_enc_num)])
            self.people_people_fc = nn.ModuleList(
                                        [nn.Sequential(
                                        nn.Linear(self.people_feat_dim, self.people_feat_dim),
                                        nn.ReLU(),
                                        nn.Linear(self.people_feat_dim, self.people_feat_dim),
                                        )
                                        for _ in range(self.people_people_trans_enc_num)
                                        ])
            self.trans_layer_norm_people_people = nn.LayerNorm(normalized_shape=self.people_feat_dim)

        if self.loss == 'mse':
            final_activation_layer = nn.Identity()
        elif self.loss == 'bce':
            final_activation_layer = nn.Sigmoid()

        if self.dataset_name == 'volleyball':
            self.rgb_cnn_extractor_type = cfg.model_params.rgb_cnn_extractor_type
            self.rgb_cnn_extractor_stage_idx = cfg.model_params.rgb_cnn_extractor_stage_idx
            if self.rgb_cnn_extractor_type == 'rgb_patch':
                down_scale_ratio = 8
            elif 'resnet' in self.rgb_cnn_extractor_type:
                self.rgb_cnn_extractor_stage_idx = self.rgb_cnn_extractor_stage_idx
                down_scale_list = [2, 4, 8, 16, 32]
                down_scale_ratio = down_scale_list[self.rgb_cnn_extractor_stage_idx]
            self.hm_height = self.resize_height//down_scale_ratio
            self.hm_width = self.resize_width//down_scale_ratio
        elif self.dataset_name == 'videocoatt':
            self.rgb_cnn_extractor_type = cfg.model_params.rgb_cnn_extractor_type
            self.rgb_cnn_extractor_stage_idx = cfg.model_params.rgb_cnn_extractor_stage_idx
            if self.rgb_cnn_extractor_type == 'rgb_patch':
                down_scale_ratio = 8
                self.hm_height = self.resize_height//down_scale_ratio
                self.hm_width = self.resize_width//down_scale_ratio
            elif 'resnet' in self.rgb_cnn_extractor_type:
                self.rgb_cnn_extractor_stage_idx = self.rgb_cnn_extractor_stage_idx
                down_scale_list = [2, 4, 8, 16, 32]
                down_scale_ratio = down_scale_list[self.rgb_cnn_extractor_stage_idx]
                self.hm_height = self.resize_height//down_scale_ratio
                self.hm_width = self.resize_width//down_scale_ratio
            elif self.rgb_cnn_extractor_type == 'davt':
                self.hm_height = 64
                self.hm_width = 64
        else:
            print('employ correct hm height and width')
            sys.exit()

        self.person_person_attention_heatmap = nn.Sequential(
            nn.Linear(self.people_feat_dim, self.people_feat_dim),
            nn.ReLU(),
            nn.Linear(self.people_feat_dim, self.people_feat_dim),
            nn.ReLU(),
            nn.Linear(self.people_feat_dim, self.hm_height*self.hm_width),
            final_activation_layer,
        )

        self.person_person_preconv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        self.person_scene_preconv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        # self.final_joint_atention_heatmap = nn.Sequential(
        #     nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=5, padding=2, stride=2, output_padding=1),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=5, padding=2, stride=2, output_padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=8, out_channels=1, kernel_size=1),
        #     final_activation_layer,
        # )

        self.final_joint_atention_heatmap = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=5, padding=2, stride=2, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=5, padding=2, stride=2, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=1),
            final_activation_layer,
        )

    def forward(self, inp):

        input_feature = inp['input_feature']
        input_gaze = inp['input_gaze']
        head_vector = inp['head_vector']
        head_feature = inp['head_feature']
        xy_axis_map = inp['xy_axis_map']
        head_xy_map = inp['head_xy_map']
        gaze_xy_map = inp['gaze_xy_map']

        # torch.autograd.set_detect_anomaly(True)
        
        # get usuful variable
        self.batch_size, people_num, _, _, _ = xy_axis_map.shape

        # position and action info handing
        head_position = torch.cat([input_feature[:, :, :2]], dim=-1)
        head_action = torch.cat([input_feature[:, :, 2:]], dim=-1)
        head_gaze = torch.cat([input_gaze[:, :, :2]], dim=-1)

        if self.use_position and self.use_gaze and self.use_action:
            head_info_params = torch.cat([head_position, head_gaze, head_action], dim=-1)
        elif self.use_position and self.use_gaze:
            head_info_params = torch.cat([head_position, head_gaze], dim=-1)
        elif self.use_position and self.use_action:
            head_info_params = torch.cat([head_position, head_action], dim=-1)
        elif self.use_gaze and self.use_action:
            head_info_params = torch.cat([head_gaze, head_action], dim=-1)
        elif self.use_position:
            head_info_params = torch.cat([head_position], dim=-1)
        elif self.use_gaze:
            head_info_params = torch.cat([head_gaze], dim=-1)
        elif self.use_action:
            head_info_params = torch.cat([head_action], dim=-1)
        else:
            print('no person information')
            sys.exit()

        # person feature embedding
        head_info_params_emb = self.head_info_feat_embeding(head_info_params)
        ja_embedding = self.ja_embedding
        ja_embedding = ja_embedding.expand(self.batch_size, 1, self.people_feat_dim)
        head_info_params_emb = torch.cat([head_info_params_emb, ja_embedding], dim=1)

        # people relation encoding
        if self.use_people_people_trans:
            key_padding_mask_people_people = (torch.sum(head_feature, dim=-1) == 0).bool()
            key_padding_mask_joint_attention = torch.zeros(self.batch_size, 1, device=head_feature.device).bool()
            key_padding_mask_people_people = torch.cat([key_padding_mask_people_people, key_padding_mask_joint_attention], dim=1)

            for i in range(self.people_people_trans_enc_num):
                head_info_params_feat, people_people_trans_weights = self.people_people_self_attention[i](head_info_params_emb, head_info_params_emb, head_info_params_emb, key_padding_mask=key_padding_mask_people_people)
                head_info_params_feat_res = head_info_params_feat + head_info_params_emb
                head_info_params_feat_feed = self.people_people_fc[i](head_info_params_feat_res)
                head_info_params_feat_feed_res = head_info_params_feat_res + head_info_params_feat_feed
                head_info_params_feat_feed_res = self.trans_layer_norm_people_people(head_info_params_feat_feed_res)
                head_info_params_emb = head_info_params_feat_feed_res

                trans_att_people_people_i = people_people_trans_weights.view(self.batch_size, 1, people_num+1, people_num+1)
                if i == 0:
                    trans_att_people_people = trans_att_people_people_i
                else:
                    trans_att_people_people = torch.cat([trans_att_people_people, trans_att_people_people_i], dim=1)
        else:
            trans_att_people_people = torch.zeros(self.batch_size, self.people_people_trans_enc_num, people_num, people_num)

        # attention estimation of person-to-person path
        attention_token = head_info_params_emb[:, :-1, :]
        person_person_attention_heatmap = self.person_person_attention_heatmap(attention_token)
        person_person_attention_heatmap = person_person_attention_heatmap.view(self.batch_size, people_num, self.hm_height, self.hm_width)
        # joint attention estimation of person-to-person path
        ja_embedding_relation = head_info_params_emb[:, -1, :]
        person_person_joint_attention_heatmap = self.person_person_attention_heatmap(ja_embedding_relation)
        person_person_joint_attention_heatmap = person_person_joint_attention_heatmap.view(self.batch_size, 1, self.hm_height, self.hm_width)

        # attention estimation of person-to-scene path
        person_scene_attention_heatmap = inp['person_scene_attention_heatmap']
        person_scene_attention_heatmap = person_scene_attention_heatmap * (torch.sum(head_feature, dim=2) != 0)[:, :, None, None]
        # joint attention estimation of person-to-scene path
        person_scene_joint_attention_heatmap = torch.sum(person_scene_attention_heatmap, dim=1)
        no_pad_idx_cnt = torch.sum((torch.sum(head_feature, dim=2) != 0), dim=1)
        person_scene_joint_attention_heatmap = person_scene_joint_attention_heatmap / no_pad_idx_cnt[:, None, None]
        person_scene_joint_attention_heatmap = person_scene_joint_attention_heatmap[:, None, :, :]

        # final joint attention estimation
        person_person_joint_attention_heatmap_preconv = self.person_person_preconv(person_person_joint_attention_heatmap)
        person_scene_joint_attention_heatmap_preconv = self.person_scene_preconv(person_scene_joint_attention_heatmap)
        dual_heatmap = torch.cat([person_person_joint_attention_heatmap_preconv, person_scene_joint_attention_heatmap_preconv], dim=1)
        final_joint_attention_heatmap = self.final_joint_atention_heatmap(dual_heatmap)

        # generate head xy map
        head_xy_map = head_xy_map * head_feature[:, :, :2, None, None]

        # generate gaze xy map
        gaze_xy_map = gaze_xy_map * head_vector[:, :, :2, None, None]

        # generate gaze cone map
        xy_axis_map_dif_head = xy_axis_map - head_xy_map
        x_axis_map_dif_head_mul_gaze = xy_axis_map_dif_head * gaze_xy_map
        xy_dot_product = torch.sum(x_axis_map_dif_head_mul_gaze, dim=2)
        xy_dot_product = xy_dot_product / (torch.norm(xy_axis_map_dif_head, dim=2) + self.epsilon)
        xy_dot_product = xy_dot_product / (torch.norm(gaze_xy_map, dim=2) + self.epsilon)

        # calculate theta and distance map
        theta_x_y = torch.acos(torch.clamp(xy_dot_product, -1+self.epsilon, 1-self.epsilon))
        distance_x_y = torch.norm(xy_axis_map_dif_head, dim=2)

        # normalize theta and distance
        theta_x_y = theta_x_y / self.pi
        distance_x_y = distance_x_y / (2**0.5)

        angle_dist = torch.exp(-torch.pow(theta_x_y, 2)/(2* 0.5 ** 2))
        angle_dist = angle_dist * (torch.sum(head_feature, dim=2) != 0)[:, :, None, None]
        distance_dist = torch.exp(-torch.pow(theta_x_y, 2)/(2* 0.5 ** 2))
        distance_dist = distance_dist * (torch.sum(head_feature, dim=2) != 0)[:, :, None, None]
        trans_att_people_rgb = torch.zeros(self.batch_size, people_num, 1, self.hm_height, self.hm_height)
        trans_att_people_rgb = torch.zeros(self.batch_size, people_num, 1, self.hm_height, self.hm_height)

        # pack return values
        data = {}
        data['person_person_attention_heatmap'] = person_person_attention_heatmap
        data['person_person_joint_attention_heatmap'] = person_person_joint_attention_heatmap
        data['person_scene_attention_heatmap'] = person_scene_attention_heatmap
        data['person_scene_joint_attention_heatmap'] = person_scene_joint_attention_heatmap
        data['final_joint_attention_heatmap'] = final_joint_attention_heatmap

        data['angle_dist'] = angle_dist
        data['distance_dist'] = distance_dist
        data['trans_att_people_rgb'] = trans_att_people_rgb
        data['trans_att_people_people'] = trans_att_people_people
        data['head_info_params'] = head_info_params

        return data

    def calc_loss(self, inp, out, cfg):
        # unpack data (input)
        img_gt_attention = inp['img_gt']
        gt_box = inp['gt_box']
        gt_box_id = inp['gt_box_id']
        att_inside_flag = inp['att_inside_flag']

        # unpack data (output)
        person_person_attention_heatmap = out['person_person_attention_heatmap']
        person_person_joint_attention_heatmap = out['person_person_joint_attention_heatmap']
        person_scene_attention_heatmap = out['person_scene_attention_heatmap']
        person_scene_joint_attention_heatmap = out['person_scene_joint_attention_heatmap']
        final_joint_attention_heatmap = out['final_joint_attention_heatmap']

        self.use_person_person_att_loss = cfg.exp_params.use_person_person_att_loss
        self.use_person_person_jo_att_loss = cfg.exp_params.use_person_person_jo_att_loss
        self.use_person_scene_att_loss = cfg.exp_params.use_person_scene_att_loss
        self.use_person_scene_jo_att_loss = cfg.exp_params.use_person_scene_jo_att_loss
        self.use_final_jo_att_loss = cfg.exp_params.use_final_jo_att_loss

        # switch loss coeficient
        if self.use_person_person_att_loss:
            loss_person_person_att_coef = 1
        else:
            loss_person_person_att_coef = 0

        if self.use_person_person_jo_att_loss:
            use_person_person_jo_att_coef = 1
        else:
            use_person_person_jo_att_coef = 0

        if self.use_person_scene_att_loss:
            use_person_scene_att_coef = 1
        else:
            use_person_scene_att_coef = 0

        if self.use_person_scene_jo_att_loss:
            use_person_scene_jo_att_coef = 1
        else:
            use_person_scene_jo_att_coef = 0
    
        if self.use_final_jo_att_loss:
            use_final_jo_att_coef = 1
        else:
            use_final_jo_att_coef = 0

        # generate gt map
        img_gt_joint_attention = torch.sum(img_gt_attention, dim=1)
        img_gt_all_thresh = torch.ones(1, device=img_gt_attention.device)
        img_gt_joint_attention = torch.where(img_gt_joint_attention>img_gt_all_thresh, img_gt_all_thresh, img_gt_joint_attention)

        # calculate person-person path loss
        person_person_attention_heatmap = F.interpolate(person_person_attention_heatmap, (self.resize_height, self.resize_width), mode='bilinear')
        loss_p_p_att = self.loss_func_hm_sum(person_person_attention_heatmap.float(), img_gt_attention.float())
        loss_p_p_att = loss_p_p_att/(torch.sum(att_inside_flag)*self.resize_height*self.resize_width)
        loss_p_p_att = loss_person_person_att_coef * loss_p_p_att
        person_person_joint_attention_heatmap = F.interpolate(person_person_joint_attention_heatmap, (self.resize_height, self.resize_width), mode='bilinear')
        person_person_joint_attention_heatmap = person_person_joint_attention_heatmap[:, 0, :, :]
        loss_p_p_jo_att = self.loss_func_hm_mean(person_person_joint_attention_heatmap.float(), img_gt_joint_attention.float())
        loss_p_p_jo_att = use_person_person_jo_att_coef * loss_p_p_jo_att
        # print('loss_p_p_att', loss_p_p_att)
        # print('loss_p_p_jo_att', loss_p_p_jo_att)

        # calculate person-scene path loss
        person_scene_attention_heatmap = F.interpolate(person_scene_attention_heatmap, (self.resize_height, self.resize_width), mode='bilinear')
        loss_p_s_att = self.loss_func_hm_sum(person_scene_attention_heatmap.float(), img_gt_attention.float())
        loss_p_s_att = loss_p_s_att/(torch.sum(att_inside_flag)*self.resize_height*self.resize_width)
        loss_p_s_att = use_person_scene_att_coef * loss_p_s_att
        person_scene_joint_attention_heatmap = F.interpolate(person_scene_joint_attention_heatmap, (self.resize_height, self.resize_width), mode='bilinear')
        person_scene_joint_attention_heatmap = person_scene_joint_attention_heatmap[:, 0, :, :]
        loss_p_s_jo_att = self.loss_func_hm_mean(person_scene_joint_attention_heatmap.float(), img_gt_joint_attention.float())
        loss_p_s_jo_att = use_person_scene_jo_att_coef * loss_p_s_jo_att
        # print('loss_p_s_att', loss_p_s_att)
        # print('loss_p_s_jo_att', loss_p_s_jo_att)

        # calculate final loss
        final_joint_attention_heatmap = F.interpolate(final_joint_attention_heatmap, (self.resize_height, self.resize_width), mode='bilinear')
        final_joint_attention_heatmap = final_joint_attention_heatmap[:, 0, :, :]
        loss_final_jo_att = self.loss_func_hm_mean(final_joint_attention_heatmap.float(), img_gt_joint_attention.float())
        loss_final_jo_att = use_final_jo_att_coef * loss_final_jo_att
        # print('loss_final_jo_att', loss_final_jo_att)

        # pack loss
        loss_set = {}
        loss_set['loss_p_p_att'] = loss_p_p_att
        loss_set['loss_p_p_jo_att'] = loss_p_p_jo_att
        loss_set['loss_p_s_att'] = loss_p_s_att
        loss_set['loss_p_s_jo_att'] = loss_p_s_jo_att
        loss_set['loss_final_jo_att'] = loss_final_jo_att

        return loss_set