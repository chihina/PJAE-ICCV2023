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
        self.p_p_estimator_type = cfg.model_params.p_p_estimator_type
        self.p_p_aggregation_type = cfg.model_params.p_p_aggregation_type

        # fusion network type
        self.fusion_net_type = cfg.model_params.fusion_net_type

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
        # self.use_final_jo_att_loss = cfg.exp_params.use_final_jo_att_loss

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

        if 'fc' in self.p_p_estimator_type:
            if self.p_p_estimator_type == 'fc_shallow':
                down_scale_ratio = 64
            elif self.p_p_estimator_type == 'fc_middle':
                down_scale_ratio = 32
            elif self.p_p_estimator_type == 'fc_deep':
                down_scale_ratio = 8
            else:
                print('please use correct p_p estimator type')
                sys.exit()
            self.hm_height = self.resize_height
            self.hm_width = self.resize_width
            self.hm_height_middle = self.resize_height//down_scale_ratio
            self.hm_width_middle = self.resize_height//down_scale_ratio
            self.person_person_attention_heatmap = nn.Sequential(
                nn.Linear(self.people_feat_dim, self.people_feat_dim),
                nn.ReLU(),
                nn.Linear(self.people_feat_dim, self.hm_height_middle*self.hm_width_middle),
                final_activation_layer,
            )
        elif 'deconv' in self.p_p_estimator_type:
            if self.p_p_estimator_type == 'deconv_shallow':
                self.latent_ch = 8
            elif self.p_p_estimator_type == 'deconv_middle':
                self.latent_ch = 64
            elif self.p_p_estimator_type == 'deconv_deep':
                self.latent_ch = 256
            else:
                print('please use correct p_p estimator type')
                sys.exit()
            self.hm_height = self.resize_height
            self.hm_width = self.resize_width
            self.hm_height_middle = self.resize_height
            self.hm_width_middle = self.resize_width
            self.latent_height = self.hm_height_middle // 64
            self.latent_width = self.hm_width_middle // 64
            self.person_person_attention_heatmap_pre_fc = nn.Sequential(
                nn.Linear(self.people_feat_dim, self.latent_height*self.latent_width*self.latent_ch),
                nn.ReLU(),
            )
            self.person_person_attention_heatmap = nn.Sequential(
                nn.ConvTranspose2d(self.latent_ch, self.latent_ch, 4, 2, 1, bias=False),
                nn.ReLU(),
                nn.ConvTranspose2d(self.latent_ch, self.latent_ch//2, 4, 2, 1, bias=False),
                nn.ReLU(),
                nn.ConvTranspose2d(self.latent_ch//2, self.latent_ch//2, 4, 2, 1, bias=False),
                nn.ReLU(),
                nn.ConvTranspose2d(self.latent_ch//2, self.latent_ch//4, 4, 2, 1, bias=False),
                nn.ReLU(),
                nn.ConvTranspose2d(self.latent_ch//4, self.latent_ch//8, 4, 2, 1, bias=False),
                nn.ReLU(),
                nn.ConvTranspose2d(self.latent_ch//8, 1, 4, 2, 1, bias=False),
                final_activation_layer,
            )
        elif 'field' in self.p_p_estimator_type:
            self.hm_height = self.resize_height
            self.hm_width = self.resize_width
            self.hm_height_middle = self.resize_height
            self.hm_width_middle = self.resize_width

            if self.p_p_estimator_type == 'field_shallow':
                self.person_person_attention_heatmap_middle = nn.Sequential(
                    nn.Identity(),
                )
            elif self.p_p_estimator_type == 'field_middle':
                self.person_person_attention_heatmap_middle = nn.Sequential(
                    nn.Linear(self.people_feat_dim, self.people_feat_dim),
                    nn.ReLU(),
                    nn.Linear(self.people_feat_dim, self.people_feat_dim),
                    nn.ReLU(),
                )
            elif self.p_p_estimator_type == 'field_deep':
                self.person_person_attention_heatmap_middle = nn.Sequential(
                    nn.Linear(self.people_feat_dim, self.people_feat_dim),
                    nn.ReLU(),
                    nn.Linear(self.people_feat_dim, self.people_feat_dim),
                    nn.ReLU(),
                    nn.Linear(self.people_feat_dim, self.people_feat_dim),
                    nn.ReLU(),
                    nn.Linear(self.people_feat_dim, self.people_feat_dim),
                    nn.ReLU(),
                )

            self.person_person_attention_heatmap = nn.Sequential(
                nn.Linear(self.people_feat_dim+2, self.people_feat_dim),
                nn.ReLU(),
                self.person_person_attention_heatmap_middle,
                nn.Linear(self.people_feat_dim, 1),
                final_activation_layer,
            )
        else:
            print('please use correct p_p estimator type')
            sys.exit()

        if self.p_p_aggregation_type == 'ind_only':
            pass
        elif self.p_p_aggregation_type == 'token_only':
            pass
        elif self.p_p_aggregation_type == 'ind_and_token_ind_based':
            pass
        elif self.p_p_aggregation_type == 'ind_and_token_token_based':
            pass
        else:
            print('please use correct p_p aggregation type')
            sys.exit()

    def forward(self, inp):

        input_feature = inp['input_feature']
        input_gaze = inp['input_gaze']
        head_vector = inp['head_vector']
        head_feature = inp['head_feature']
        xy_axis_map = inp['xy_axis_map']
        head_xy_map = inp['head_xy_map']
        gaze_xy_map = inp['gaze_xy_map']
        att_inside_flag = inp['att_inside_flag']

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
        if self.p_p_aggregation_type == 'ind_only':
            attention_token = head_info_params_emb[:, :-1, :]
        elif self.p_p_aggregation_type == 'token_only':
            attention_token = head_info_params_emb[:, :-1, :]
        elif self.p_p_aggregation_type == 'ind_and_token_ind_based':
            attention_token = head_info_params_emb[:, :-1, :]
            ja_embedding_relation = head_info_params_emb[:, -1, :][:, None, :]
            attention_token = attention_token + head_info_params_emb[:, -1, :][:, None, :]
        elif self.p_p_aggregation_type == 'ind_and_token_ind_based':
            attention_token = head_info_params_emb[:, :-1, :]
        elif self.p_p_aggregation_type == 'ind_and_token_token_based':
            attention_token = head_info_params_emb[:, :-1, :]
        else:
            sys.exit()

        if 'fc' in self.p_p_estimator_type:
            attention_token_input = attention_token.view(self.batch_size, people_num, self.people_feat_dim)
        elif 'deconv' in self.p_p_estimator_type:
            attention_token_input = attention_token.view(self.batch_size, people_num, self.people_feat_dim)
            attention_token_input = self.person_person_attention_heatmap_pre_fc(attention_token_input)
            attention_token_input = attention_token_input.view(self.batch_size*people_num, self.latent_ch, self.latent_height, self.latent_width)
        elif 'field' in self.p_p_estimator_type:
            attention_token_view = attention_token.view(self.batch_size, people_num, 1, self.people_feat_dim)
            attention_token_expand = attention_token_view.expand(self.batch_size, people_num, self.hm_height_middle*self.hm_width_middle, self.people_feat_dim)
            x_axis_map = xy_axis_map[:, :, 0, :, :]
            y_axis_map = xy_axis_map[:, :, 1, :, :]
            x_axis_map = F.interpolate(x_axis_map, (self.hm_height_middle, self.hm_width_middle), mode='bilinear')
            y_axis_map = F.interpolate(y_axis_map, (self.hm_height_middle, self.hm_width_middle), mode='bilinear')
            x_axis_map = x_axis_map.view(self.batch_size, people_num, self.hm_height_middle*self.hm_width_middle, 1)
            y_axis_map = y_axis_map.view(self.batch_size, people_num, self.hm_height_middle*self.hm_width_middle, 1)
            attention_token_input = torch.cat([attention_token_expand, x_axis_map, y_axis_map], dim=-1)
        person_person_attention_heatmap = self.person_person_attention_heatmap(attention_token_input)
        person_person_attention_heatmap = person_person_attention_heatmap.view(self.batch_size, people_num, self.hm_height_middle, self.hm_width_middle)
        person_person_attention_heatmap = F.interpolate(person_person_attention_heatmap, (self.hm_height, self.hm_width), mode='bilinear')

        # joint attention estimation of person-to-person path
        ja_embedding_relation = head_info_params_emb[:, -1, :]
        if 'fc' in self.p_p_estimator_type:
            ja_embedding_relation_input = ja_embedding_relation.view(self.batch_size, 1, self.people_feat_dim)
        elif 'deconv' in self.p_p_estimator_type:
            ja_embedding_relation_input = ja_embedding_relation.view(self.batch_size, 1, self.people_feat_dim)
            ja_embedding_relation_input = self.person_person_attention_heatmap_pre_fc(ja_embedding_relation_input)
            ja_embedding_relation_input = ja_embedding_relation_input.view(self.batch_size, self.latent_ch, self.latent_height, self.latent_width)
        elif 'field' in self.p_p_estimator_type:
            ja_embedding_relation_view = ja_embedding_relation.view(self.batch_size, 1, 1, self.people_feat_dim)
            ja_embedding_relation_expand = ja_embedding_relation_view.expand(self.batch_size, 1, self.hm_height_middle*self.hm_width_middle, self.people_feat_dim)
            x_axis_map = xy_axis_map[:, 0, 0, :, :][:, None, :, :]
            y_axis_map = xy_axis_map[:, 0, 1, :, :][:, None, :, :]
            x_axis_map = F.interpolate(x_axis_map, (self.hm_height_middle, self.hm_width_middle), mode='bilinear')
            y_axis_map = F.interpolate(y_axis_map, (self.hm_height_middle, self.hm_width_middle), mode='bilinear')
            x_axis_map = x_axis_map.view(self.batch_size, 1, self.hm_height_middle*self.hm_width_middle, 1)
            y_axis_map = y_axis_map.view(self.batch_size, 1, self.hm_height_middle*self.hm_width_middle, 1)
            ja_embedding_relation_input = torch.cat([ja_embedding_relation_expand, x_axis_map, y_axis_map], dim=-1)
        person_person_joint_attention_heatmap = self.person_person_attention_heatmap(ja_embedding_relation_input)
        person_person_joint_attention_heatmap = person_person_joint_attention_heatmap.view(self.batch_size, 1, self.hm_height_middle, self.hm_width_middle)
        person_person_joint_attention_heatmap = F.interpolate(person_person_joint_attention_heatmap, (self.hm_height, self.hm_width), mode='bilinear')

        # final p_p heatmap aggregation
        if self.p_p_aggregation_type == 'ind_only':
            person_person_attention_heatmap = person_person_attention_heatmap * att_inside_flag[:, :, None, None]
            person_person_joint_attention_heatmap = torch.sum(person_person_attention_heatmap, dim=1)
            person_person_joint_attention_heatmap = person_person_joint_attention_heatmap / torch.sum(att_inside_flag, dim=1)[:, None, None]
            person_person_joint_attention_heatmap = person_person_joint_attention_heatmap[:, None, :, :]
        elif self.p_p_aggregation_type == 'token_only':
            pass
        elif self.p_p_aggregation_type == 'ind_and_token_ind_based':
            person_person_attention_heatmap = person_person_attention_heatmap * att_inside_flag[:, :, None, None]
            person_person_joint_attention_heatmap = torch.sum(person_person_attention_heatmap, dim=1)
            person_person_joint_attention_heatmap = person_person_joint_attention_heatmap / torch.sum(att_inside_flag, dim=1)[:, None, None]
            person_person_joint_attention_heatmap = person_person_joint_attention_heatmap[:, None, :, :]
        elif self.p_p_aggregation_type == 'ind_and_token_token_based':
            pass
        else:
            sys.exit()

        # attention estimation of person-to-scene path
        person_scene_attention_heatmap = inp['person_scene_attention_heatmap']
        person_scene_attention_heatmap = person_scene_attention_heatmap.view(self.batch_size, people_num, self.hm_height_middle_p_s, self.hm_width_middle_p_s)
        person_scene_attention_heatmap = F.interpolate(person_scene_attention_heatmap, (self.hm_height_p_s, self.hm_width_p_s), mode='bilinear')
        person_scene_attention_heatmap = person_scene_attention_heatmap * (torch.sum(head_feature, dim=2) != 0)[:, :, None, None]

        # joint attention estimation of person-to-scene path
        person_scene_joint_attention_heatmap = torch.sum(person_scene_attention_heatmap, dim=1)
        no_pad_idx_cnt = torch.sum((torch.sum(head_feature, dim=2) != 0), dim=1)
        person_scene_joint_attention_heatmap = person_scene_joint_attention_heatmap / no_pad_idx_cnt[:, None, None]
        person_scene_joint_attention_heatmap = person_scene_joint_attention_heatmap[:, None, :, :]

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

        self.use_person_person_att_loss = cfg.exp_params.use_person_person_att_loss
        self.person_person_att_loss_weight = cfg.exp_params.person_person_att_loss_weight

        self.use_person_person_jo_att_loss = cfg.exp_params.use_person_person_jo_att_loss
        self.person_person_jo_att_loss_weight = cfg.exp_params.person_person_jo_att_loss_weight
        
        self.use_person_scene_att_loss = cfg.exp_params.use_person_scene_att_loss
        self.person_scene_att_loss_weight = cfg.exp_params.person_scene_att_loss_weight
        
        self.use_person_scene_jo_att_loss = cfg.exp_params.use_person_scene_jo_att_loss
        self.person_scene_jo_att_loss_weight = cfg.exp_params.person_scene_jo_att_loss_weight
        
        # self.use_final_jo_att_loss = cfg.exp_params.use_final_jo_att_loss
        # self.final_jo_att_loss_weight = cfg.exp_params.final_jo_att_loss_weight

        # switch loss coeficient
        if self.use_person_person_att_loss:
            loss_person_person_att_coef = self.person_person_att_loss_weight
        else:
            loss_person_person_att_coef = 0

        if self.use_person_person_jo_att_loss:
            use_person_person_jo_att_coef = self.person_person_jo_att_loss_weight
        else:
            use_person_person_jo_att_coef = 0

        if self.use_person_scene_att_loss:
            use_person_scene_att_coef = self.person_scene_att_loss_weight
        else:
            use_person_scene_att_coef = 0

        if self.use_person_scene_jo_att_loss:
            use_person_scene_jo_att_coef = self.person_scene_jo_att_loss_weight
        else:
            use_person_scene_jo_att_coef = 0
    
        # if self.use_final_jo_att_loss:
        #     use_final_jo_att_coef = self.final_jo_att_loss_weight
        # else:
        #     use_final_jo_att_coef = 0

        # generate gt map
        img_gt_attention = (img_gt_attention * att_inside_flag[:, :, None, None]).float()
        img_gt_joint_attention = torch.sum(img_gt_attention, dim=1)
        img_gt_all_thresh = torch.ones(1, device=img_gt_attention.device).float()
        img_gt_joint_attention = torch.where(img_gt_joint_attention>=img_gt_all_thresh, img_gt_all_thresh, img_gt_joint_attention)

        # calculate person-person path loss
        person_person_attention_heatmap = F.interpolate(person_person_attention_heatmap, (self.resize_height, self.resize_width), mode='bilinear')
        person_person_attention_heatmap = person_person_attention_heatmap * att_inside_flag[:, :, None, None]
        loss_p_p_att = self.loss_func_hm_sum(person_person_attention_heatmap.float(), img_gt_attention.float())
        loss_p_p_att = loss_p_p_att/(torch.sum(att_inside_flag)*self.resize_height*self.resize_width)
        loss_p_p_att = loss_person_person_att_coef * loss_p_p_att
        person_person_joint_attention_heatmap = F.interpolate(person_person_joint_attention_heatmap, (self.resize_height, self.resize_width), mode='bilinear')
        person_person_joint_attention_heatmap = person_person_joint_attention_heatmap[:, 0, :, :]
        loss_p_p_jo_att = self.loss_func_hm_mean(person_person_joint_attention_heatmap.float(), img_gt_joint_attention.float())
        loss_p_p_jo_att = use_person_person_jo_att_coef * loss_p_p_jo_att
        # print('loss_p_p_att', loss_p_p_att)
        # print('loss_p_p_jo_att', loss_p_p_jo_att)
        # print('JA', f'{torch.min(person_person_joint_attention_heatmap).item():.2f} {torch.max(person_person_joint_attention_heatmap).item():.2f}')
        # print('AT', f'{torch.min(person_scene_joint_attention_heatmap).item():.2f} {torch.max(person_scene_joint_attention_heatmap).item():.2f}')
        # print('GT', f'{torch.min(img_gt_joint_attention).item():.2f} {torch.max(img_gt_joint_attention).item():.2f}')
        # print('F', f'{torch.min(final_joint_attention_heatmap).item():.2f} {torch.max(final_joint_attention_heatmap).item():.2f}')

        # calculate person-scene path loss
        person_scene_attention_heatmap = F.interpolate(person_scene_attention_heatmap, (self.resize_height, self.resize_width), mode='bilinear')
        person_scene_attention_heatmap = person_scene_attention_heatmap * att_inside_flag[:, :, None, None]
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
        # final_joint_attention_heatmap = F.interpolate(final_joint_attention_heatmap, (self.resize_height, self.resize_width), mode='bilinear')
        # final_joint_attention_heatmap = final_joint_attention_heatmap[:, 0, :, :]
        # loss_final_jo_att = self.loss_func_hm_mean(final_joint_attention_heatmap.float(), img_gt_joint_attention.float())
        # loss_final_jo_att = use_final_jo_att_coef * loss_final_jo_att
        # print('loss_final_jo_att', loss_final_jo_att)

        # pack loss
        loss_set = {}
        loss_set['loss_p_p_att'] = loss_p_p_att
        loss_set['loss_p_p_jo_att'] = loss_p_p_jo_att
        loss_set['loss_p_s_att'] = loss_p_s_att
        loss_set['loss_p_s_jo_att'] = loss_p_s_jo_att

        return loss_set