import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision
import torchvision.models as models
from models.model_utils import positionalencoding1d, positionalencoding2d
from roi_align.roi_align import RoIAlign

import sys
import  numpy as np
import cv2

class JointAttentionEstimatorTransformerDualOnlyPeopleImgFeat(nn.Module):
    def __init__(self, cfg):
        super(JointAttentionEstimatorTransformerDualOnlyPeopleImgFeat, self).__init__()

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
        self.use_position = cfg.model_params.use_position
        self.use_gaze = cfg.model_params.use_gaze
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

        # attribute prediction loss
        self.use_ind_feat_crop = 'crop_single'
        if 'use_ind_feat_crop' in cfg.model_params:
            self.use_ind_feat_crop = cfg.model_params.use_ind_feat_crop
        self.use_action_loss = cfg.model_params.use_action_loss
        self.use_action_class_num = cfg.model_params.use_action_class_num
        self.action_loss_coef = cfg.model_params.action_loss_coef
        self.use_gaze_loss = cfg.model_params.use_gaze_loss
        self.gaze_loss_coef = cfg.model_params.gaze_loss_coef
        self.gaze_loss_type = cfg.model_params.gaze_loss_type
        self.use_attribute_loss_type = cfg.model_params.use_attribute_loss_type

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

        self.use_frame_type = cfg.exp_params.use_frame_type
        self.temporal_fusion_type = cfg.model_params.temporal_fusion_type
        if 'atb' in self.temporal_fusion_type:
            self.mha_num_heads_atb = cfg.model_params.mha_num_heads_atb
            self.atb_trans_enc_num = cfg.model_params.atb_trans_enc_num
        if 'token' in self.temporal_fusion_type:
            self.ja_token_trans_enc_num = cfg.model_params.ja_token_trans_enc_num
            self.mha_num_heads_ja_token = cfg.model_params.mha_num_heads_ja_token
            self.ja_token_tem_enc = positionalencoding1d(self.people_feat_dim, 100)            

        if self.use_ind_feat_crop == 'roi_multi':
            self.crop_size = 5, 5
            self.K = self.crop_size[0]
            self.D = 512
            self.out_size = 10, 20
            self.backbone = models.vgg19(pretrained=True)
            self.backbone = self.backbone.features
            self.roi_align = RoIAlign(*self.crop_size)
            self.fc_emb_1 = nn.Linear(self.K * self.K * self.D, self.people_feat_dim)
            self.nl_emb_1 = nn.LayerNorm([self.people_feat_dim])
        elif self.use_ind_feat_crop == 'crop_single':
            self.crop_person_feat_extractor = models.vgg19(pretrained=True)
            self.crop_person_feat_extractor.classifier = self.crop_person_feat_extractor.classifier[:-1]
            self.person_feat_encoder = nn.Sequential(
                nn.Linear(4096, self.people_feat_dim),
                nn.ReLU(),
                nn.Dropout(p=0.5),
            )
        if self.use_action:
            self.action_loss_weight = [1., 1., 2., 3., 1., 2., 2., 0.2, 1.]
            self.action_loss_weight = torch.tensor(self.action_loss_weight).to(self.device)

            if self.use_frame_type == 'all':
                self.action_transformer = nn.TransformerEncoderLayer(d_model=self.people_feat_dim, nhead=2)
            
            self.action_predictor_head = nn.Sequential(
                nn.Linear(self.people_feat_dim, self.people_feat_dim),
                nn.ReLU(),
                nn.Linear(self.people_feat_dim, self.use_action_class_num),
            )

        if self.use_gaze_loss:
            if self.gaze_loss_type == 'whole':
                self.gaze_predictor_feat_ext = nn.Sequential(
                    nn.Linear(self.people_feat_dim, self.people_feat_dim),
                    nn.ReLU(),
                    nn.Linear(self.people_feat_dim, self.people_feat_dim),
                    nn.ReLU(),
                )

            elif self.gaze_loss_type == 'head':
                resnet = models.resnet18(pretrained=True)
                resnet = nn.Sequential(*list(resnet.children())[:-1])
                self.gaze_predictor_feat_ext = nn.Sequential(
                    resnet,
                    nn.Flatten(),
                    nn.Linear(512, self.people_feat_dim),
                    nn.ReLU(),
                )

            if self.use_frame_type == 'all':
                self.gaze_transformer = nn.TransformerEncoderLayer(d_model=self.people_feat_dim, nhead=2)
            
            self.gaze_predictor_head = nn.Sequential(
                nn.Linear(self.people_feat_dim, self.people_feat_dim),
                nn.ReLU(),
                nn.Linear(self.people_feat_dim, 2),
            )

        embeding_param_num = 2
        if self.use_action:
            embeding_param_num += self.use_action_class_num
        if self.use_gaze:
            embeding_param_num += 2
        self.head_info_feat_embeding = nn.Sequential(
            nn.Linear(embeding_param_num, self.people_feat_dim),
            nn.ReLU(),
            nn.Linear(self.people_feat_dim, self.people_feat_dim),
            nn.ReLU(),
            nn.Linear(self.people_feat_dim, self.people_feat_dim),
        )

        if self.use_frame_type == 'all' and 'atb' in self.temporal_fusion_type:
            self.atb_feat_embeding = nn.Sequential(
                nn.Linear(self.people_feat_dim, self.people_feat_dim),
                nn.ReLU(),
                nn.Linear(self.people_feat_dim, self.people_feat_dim),
                nn.ReLU(),
                nn.Linear(self.people_feat_dim, self.people_feat_dim),
            )

            # spatial branch
            self.atb_self_att_spa = nn.ModuleList([nn.MultiheadAttention(embed_dim=self.people_feat_dim, num_heads=self.mha_num_heads_atb, batch_first=True) for _ in range(self.atb_trans_enc_num)])
            self.atb_fc_spa = nn.ModuleList(
                                        [nn.Sequential(
                                        nn.Linear(self.people_feat_dim, self.people_feat_dim),
                                        nn.ReLU(),
                                        nn.Linear(self.people_feat_dim, self.people_feat_dim),
                                        )
                                        for _ in range(self.atb_trans_enc_num)
                                        ])
            self.trans_layer_norm_atb_spa = nn.LayerNorm(normalized_shape=self.people_feat_dim)

            # temporal branch
            self.atb_self_att_tem = nn.ModuleList([nn.MultiheadAttention(embed_dim=self.people_feat_dim, num_heads=self.mha_num_heads_atb, batch_first=True) for _ in range(self.atb_trans_enc_num)])
            self.atb_fc_tem = nn.ModuleList(
                                        [nn.Sequential(
                                        nn.Linear(self.people_feat_dim, self.people_feat_dim),
                                        nn.ReLU(),
                                        nn.Linear(self.people_feat_dim, self.people_feat_dim),
                                        )
                                        for _ in range(self.atb_trans_enc_num)
                                        ])
            self.trans_layer_norm_atb_tem = nn.LayerNorm(normalized_shape=self.people_feat_dim)
        
            if self.use_action:
                self.action_predictor_refiner = nn.Sequential(
                    nn.Linear(self.people_feat_dim, self.people_feat_dim),
                    nn.ReLU(),
                    nn.Linear(self.people_feat_dim, self.use_action_class_num),
                )
            
            if self.use_gaze:
                self.gaze_predictor_refiner = nn.Sequential(
                    nn.Linear(self.people_feat_dim, self.people_feat_dim),
                    nn.ReLU(),
                    nn.Linear(self.people_feat_dim, 2),
                )

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

        if self.use_frame_type == 'all' and 'token' in self.temporal_fusion_type:
            self.ja_embedding_self_attention = nn.ModuleList([nn.MultiheadAttention(embed_dim=self.people_feat_dim, num_heads=self.mha_num_heads_ja_token, batch_first=True) for _ in range(self.ja_token_trans_enc_num)])
            self.ja_embedding_fc = nn.ModuleList(
                                        [nn.Sequential(
                                        nn.Linear(self.people_feat_dim, self.people_feat_dim),
                                        nn.ReLU(),
                                        nn.Linear(self.people_feat_dim, self.people_feat_dim),
                                        )
                                        for _ in range(self.people_people_trans_enc_num)
                                        ])
            self.trans_layer_norm_ja_embedding = nn.LayerNorm(normalized_shape=self.people_feat_dim)

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
        elif self.p_p_aggregation_type == 'token_only_concat':
            self.person_person_attention_heatmap_middle_concat = nn.Sequential(
                nn.Linear(self.people_feat_dim*12, self.people_feat_dim*12),
                nn.ReLU(),
                nn.Linear(self.people_feat_dim*12, self.people_feat_dim*12),
                nn.ReLU(),
            )

            self.person_person_attention_heatmap_concat = nn.Sequential(
                nn.Linear(self.people_feat_dim*12+2, self.people_feat_dim*12),
                nn.ReLU(),
                self.person_person_attention_heatmap_middle_concat,
                nn.Linear(self.people_feat_dim*12, 1),
                final_activation_layer,
            )
        elif self.p_p_aggregation_type == 'ind_and_token_ind_based':
            pass
        elif self.p_p_aggregation_type == 'ind_and_token_token_based':
            pass
        else:
            print('please use correct p_p aggregation type')
            # sys.exit()

    def forward(self, inp):

        input_feature = inp['input_feature']
        input_gaze = inp['input_gaze']
        head_vector = inp['head_vector']
        head_feature = inp['head_feature']
        xy_axis_map = inp['xy_axis_map']
        head_xy_map = inp['head_xy_map']
        gaze_xy_map = inp['gaze_xy_map']
        att_inside_flag = inp['att_inside_flag']
        people_bbox = inp['people_bbox']
        people_bbox_norm = inp['people_bbox_norm']

        # get usuful variable
        self.batch_size, self.frame_num, self.people_num, _ = input_feature.shape

        # get preprocessed attributes
        head_position = torch.cat([input_feature[:, :, :, :2]], dim=-1)
        head_action = torch.cat([input_feature[:, :, :, 2:]], dim=-1)
        head_gaze = torch.cat([input_gaze[:, :, :, :2]], dim=-1)

        # if prepare attributes
        if self.use_attribute_loss_type == 'interm':
            if self.use_ind_feat_crop in ['roi_multi']:
                rgb_img = inp['rgb_img']
                rgb_img = rgb_img.view(self.batch_size*self.frame_num, 3, self.resize_height, self.resize_width)
                outputs = self.backbone(rgb_img).unsqueeze(0)

                # Build features
                OH, OW = self.out_size
                features_multiscale = []
                for features in outputs:
                    if features.shape[2:4] != torch.Size([OH, OW]):
                        features = F.interpolate(features, size=(OH, OW), mode='bilinear', align_corners=True)
                    features_multiscale.append(features)
                features_multiscale = torch.cat(features_multiscale, dim=1)

                # RoI Align
                boxes_in_flat = people_bbox.view(self.batch_size*self.frame_num*self.people_num, 4)
                boxes_idx = [i * torch.ones(self.frame_num*self.people_num, dtype=torch.int) for i in range(self.batch_size)]
                boxes_idx = torch.stack(boxes_idx).to(device=people_bbox.device)
                boxes_idx_flat = torch.reshape(boxes_idx, (self.batch_size*self.frame_num*self.people_num,))
                boxes_in_flat.requires_grad = False
                boxes_idx_flat.requires_grad = False
                boxes_features = self.roi_align(features_multiscale,
                                                boxes_in_flat,
                                                boxes_idx_flat)
                boxes_features = boxes_features.reshape(self.batch_size, self.frame_num, self.people_num, -1)

                # Embedding
                boxes_features_emb = self.fc_emb_1(boxes_features)
                boxes_features_emb = self.nl_emb_1(boxes_features_emb)
                ind_app_feat = F.relu(boxes_features_emb)
            elif self.use_ind_feat_crop in ['crop_single']:
                rgb_img_person = inp['rgb_img_person']
                _, _, _, img_channel, img_height_person, img_width_person  = rgb_img_person.shape
                rgb_img_person = rgb_img_person.view(self.batch_size*self.frame_num*self.people_num, img_channel, img_height_person, img_width_person)
                crop_person_feat = self.crop_person_feat_extractor(rgb_img_person)
                crop_person_feat = crop_person_feat.view(self.batch_size*self.frame_num*self.people_num, -1)
                ind_app_feat = self.person_feat_encoder(crop_person_feat)
                ind_app_feat = ind_app_feat.view(self.batch_size, self.frame_num, self.people_num, self.people_feat_dim)
            else:
                assert False, 'use_ind_feat_crop error'
            
            if self.use_action:
                ind_app_feat_trans = ind_app_feat.transpose(2, 1).reshape(self.batch_size*self.people_num, self.frame_num, self.people_feat_dim) 
                if self.use_frame_type == 'all':
                    ind_app_feat_trans = self.action_transformer(ind_app_feat_trans)
                ind_app_feat_trans = ind_app_feat_trans.reshape(self.batch_size, self.people_num, self.frame_num, self.people_feat_dim)
                ind_app_feat_trans = ind_app_feat_trans.transpose(2, 1)
                action_person_predicted = self.action_predictor_head(ind_app_feat_trans)
                head_action = action_person_predicted.view(self.batch_size, self.frame_num, self.people_num, self.use_action_class_num)
            if self.use_gaze:
                if self.gaze_loss_type == 'whole':
                    gaze_person_feature = self.gaze_predictor_feat_ext(ind_app_feat)
                elif self.gaze_loss_type == 'head':
                    head_img = inp['head_img']
                    _, _, _, _, head_img_height, head_img_width  = head_img.shape
                    head_img_flat = head_img.view(self.batch_size*self.frame_num*self.people_num, 3, head_img_height, head_img_width)
                    gaze_person_feature = self.gaze_predictor_feat_ext(head_img_flat)
                    gaze_person_feature = gaze_person_feature.view(self.batch_size, self.frame_num, self.people_num, self.people_feat_dim)

                gaze_person_feature = gaze_person_feature.transpose(2, 1).reshape(self.batch_size*self.people_num, self.frame_num, self.people_feat_dim)
                if self.use_frame_type == 'all':
                    gaze_person_feature = self.gaze_transformer(gaze_person_feature)
                gaze_person_feature = gaze_person_feature.reshape(self.batch_size, self.people_num, self.frame_num, self.people_feat_dim)
                gaze_person_feature = gaze_person_feature.transpose(2, 1)
                gaze_person_predicted = self.gaze_predictor_head(gaze_person_feature)
                gaze_person_predicted = F.normalize(gaze_person_predicted, dim=-1)
                head_gaze = gaze_person_predicted.view(self.batch_size, self.frame_num, self.people_num, 2)

        if self.use_frame_type == 'all' and 'atb' in self.temporal_fusion_type:
            # feature extraction
            if self.use_action and self.use_gaze:
                head_info_params_atb = torch.cat([head_position, head_action, head_gaze], dim=-1)
            elif self.use_action:
                head_info_params_atb = torch.cat([head_position, head_action], dim=-1)
            elif self.use_gaze:
                head_info_params_atb = torch.cat([head_position, head_gaze], dim=-1)
            else:
                assert False, 'use_action_loss and use_gaze_loss error'
            head_info_params_emb_atb = self.head_info_feat_embeding(head_info_params_atb)

            # temporal branch
            head_info_params_emb_atb = head_info_params_emb_atb.transpose(2, 1).reshape(self.batch_size*self.people_num, self.frame_num, self.people_feat_dim)
            for i in range(self.atb_trans_enc_num):
                head_info_params_feat, attribute_temporal_weights = self.atb_self_att_tem[i](head_info_params_emb_atb, head_info_params_emb_atb, head_info_params_emb_atb)
                head_info_params_feat_res = head_info_params_feat + head_info_params_emb_atb
                head_info_params_feat_feed = self.atb_fc_tem[i](head_info_params_feat_res)
                head_info_params_feat_feed_res = head_info_params_feat_res + head_info_params_feat_feed
                head_info_params_feat_feed = self.trans_layer_norm_atb_tem(head_info_params_feat_feed_res)
                head_info_params_emb_atb = head_info_params_feat_feed_res
            head_info_params_emb_atb = head_info_params_emb_atb.reshape(self.batch_size, self.people_num, self.frame_num, self.people_feat_dim)
            head_info_params_emb_atb = head_info_params_emb_atb.transpose(2, 1)

            # spatial branch
            head_info_params_emb_atb = head_info_params_emb_atb.reshape(self.batch_size*self.frame_num, self.people_num, self.people_feat_dim)
            key_padding_mask_people_people = (torch.sum(people_bbox, dim=-1) == 0).bool()
            key_padding_mask_people_people = key_padding_mask_people_people.view(self.batch_size*self.frame_num, self.people_num)
            for i in range(self.atb_trans_enc_num):
                head_info_params_feat, attribute_spatial_weights = self.atb_self_att_spa[i](head_info_params_emb_atb, head_info_params_emb_atb, head_info_params_emb_atb, key_padding_mask=key_padding_mask_people_people)
                head_info_params_feat_res = head_info_params_feat + head_info_params_emb_atb
                head_info_params_feat_feed = self.atb_fc_spa[i](head_info_params_feat_res)
                head_info_params_feat_feed_res = head_info_params_feat_res + head_info_params_feat_feed
                head_info_params_feat_feed = self.trans_layer_norm_atb_spa(head_info_params_feat_feed_res)
                head_info_params_emb_atb = head_info_params_feat_feed_res
            head_info_params_emb_atb = head_info_params_emb_atb.view(self.batch_size, self.frame_num, self.people_num, self.people_feat_dim)

            # attribute refinement
            if self.use_action:
                action_person_predicted_update = self.action_predictor_refiner(head_info_params_emb_atb)
            if self.use_gaze:
                gaze_person_predicted_update = F.normalize(self.gaze_predictor_refiner(head_info_params_emb_atb), dim=-1)
            
            if self.use_action and self.use_gaze:
                head_info_params = torch.cat([head_position, action_person_predicted_update, gaze_person_predicted_update], dim=-1)
            elif self.use_action:
                head_info_params = torch.cat([head_position, action_person_predicted_update], dim=-1)
            elif self.use_gaze:
                head_info_params = torch.cat([head_position, gaze_person_predicted_update], dim=-1)
        else:
            if self.use_action and self.use_gaze:
                head_info_params = torch.cat([head_position, head_action, head_gaze], dim=-1)
            elif self.use_action:
                head_info_params = torch.cat([head_position, head_action], dim=-1)
            elif self.use_gaze:
                head_info_params = torch.cat([head_position, head_gaze], dim=-1)
            else:
                assert False, 'use_action_loss and use_gaze_loss error'

        head_info_params_emb = self.head_info_feat_embeding(head_info_params)
        ja_embedding = self.ja_embedding
        ja_embedding = ja_embedding.expand(self.batch_size, self.frame_num, 1, self.people_feat_dim)
        head_info_params_emb = torch.cat([head_info_params_emb, ja_embedding], dim=-2)
        head_info_params_emb = head_info_params_emb.view(self.batch_size*self.frame_num, self.people_num+1, self.people_feat_dim)

        # people relation encoding
        if self.use_people_people_trans:
            key_padding_mask_people_people = (torch.sum(people_bbox, dim=-1) == 0).bool()
            key_padding_mask_joint_attention = torch.zeros(self.batch_size, self.frame_num, 1, device=head_feature.device).bool()
            key_padding_mask_people_people = torch.cat([key_padding_mask_people_people, key_padding_mask_joint_attention], dim=-1)
            key_padding_mask_people_people = key_padding_mask_people_people.view(self.batch_size*self.frame_num, self.people_num+1)

            for i in range(self.people_people_trans_enc_num):
                head_info_params_feat, people_people_trans_weights = self.people_people_self_attention[i](head_info_params_emb, head_info_params_emb, head_info_params_emb, key_padding_mask=key_padding_mask_people_people)
                head_info_params_feat_res = head_info_params_feat + head_info_params_emb
                head_info_params_feat_feed = self.people_people_fc[i](head_info_params_feat_res)
                head_info_params_feat_feed_res = head_info_params_feat_res + head_info_params_feat_feed
                head_info_params_feat_feed_res = self.trans_layer_norm_people_people(head_info_params_feat_feed_res)
                head_info_params_emb = head_info_params_feat_feed_res

                # trans_att_people_people_i = people_people_trans_weights.view(self.batch_size, 1, self.people_num+1, self.people_num+1)
                trans_att_people_people_i = people_people_trans_weights.view(self.batch_size, self.frame_num, 1, self.people_num+1, self.people_num+1)
                if i == 0:
                    trans_att_people_people = trans_att_people_people_i
                else:
                    trans_att_people_people = torch.cat([trans_att_people_people, trans_att_people_people_i], dim=2)
        else:
            # trans_att_people_people = torch.zeros(self.batch_size, self.people_people_trans_enc_num, self.people_num, self.people_num)
            trans_att_people_people = torch.zeros(self.batch_size, self.frame_num, self.people_people_trans_enc_num, self.people_num, self.people_num)
        head_info_params_emb = head_info_params_emb.view(self.batch_size, self.frame_num, self.people_num+1, self.people_feat_dim)

        # attention estimation of person-to-person path
        if self.use_person_person_att_loss:
            if self.p_p_aggregation_type == 'ind_only':
                attention_token = head_info_params_emb[:, :, :-1, :]
            elif self.p_p_aggregation_type == 'token_only':
                attention_token = head_info_params_emb[:, :, :-1, :]
            elif self.p_p_aggregation_type == 'token_only_concat':
                attention_token = head_info_params_emb[:, :, :-1, :]
            elif self.p_p_aggregation_type == 'ind_and_token_ind_based':
                attention_token = head_info_params_emb[:, :, :-1, :]
                ja_embedding_relation = head_info_params_emb[:, :, -1, :][:, :, None, :]
                attention_token = attention_token + head_info_params_emb[:, :, -1, :][:, :, None, :]
            elif self.p_p_aggregation_type == 'ind_and_token_ind_based':
                attention_token = head_info_params_emb[:, :, :-1, :]
            elif self.p_p_aggregation_type == 'ind_and_token_token_based':
                attention_token = head_info_params_emb[:, :, :-1, :]
            else:
                attention_token = head_info_params_emb[:, :, :-1, :]
            
            if 'fc' in self.p_p_estimator_type:
                pass
            elif 'deconv' in self.p_p_estimator_type:
                attention_token_input = self.person_person_attention_heatmap_pre_fc(attention_token_input)
                attention_token_input = attention_token_input.view(self.batch_size*self.frame_num*self.people_num, self.latent_ch, self.latent_height, self.latent_width)
            elif 'field' in self.p_p_estimator_type:
                attention_token_view = attention_token.view(self.batch_size, self.frame_num, self.people_num, 1, self.people_feat_dim)
                attention_token_expand = attention_token_view.expand(self.batch_size, self.frame_num, self.people_num, self.hm_height_middle*self.hm_width_middle, self.people_feat_dim)
                xy_axis_map = xy_axis_map.view(self.batch_size, 1, self.people_num, 2, self.hm_height_middle, self.hm_width_middle)
                xy_axis_map = xy_axis_map.expand(self.batch_size, self.frame_num, self.people_num, 2, self.hm_height_middle, self.hm_width_middle)
                x_axis_map = xy_axis_map[:, :, :, 0, :, :]
                y_axis_map = xy_axis_map[:, :, :, 1, :, :]
                x_axis_map = x_axis_map.view(self.batch_size, self.frame_num, self.people_num, self.hm_height_middle*self.hm_width_middle, 1)
                y_axis_map = y_axis_map.view(self.batch_size, self.frame_num, self.people_num, self.hm_height_middle*self.hm_width_middle, 1)
                attention_token_input = torch.cat([attention_token_expand, x_axis_map, y_axis_map], dim=-1)

            person_person_attention_heatmap = self.person_person_attention_heatmap(attention_token_input)
            person_person_attention_heatmap = person_person_attention_heatmap.view(self.batch_size*self.frame_num, self.people_num, self.hm_height_middle, self.hm_width_middle)
            person_person_attention_heatmap = F.interpolate(person_person_attention_heatmap, (self.hm_height, self.hm_width), mode='bilinear')
            person_person_attention_heatmap = person_person_attention_heatmap.view(self.batch_size, self.frame_num, self.people_num, self.hm_height, self.hm_width)
        else:
            person_person_attention_heatmap = torch.zeros(self.batch_size, self.frame_num, self.people_num, self.hm_height, self.hm_width, device=head_feature.device)
        
        # joint attention estimation of person-to-person path
        ja_embedding_relation = head_info_params_emb[:, :, -1, :]
        if self.use_frame_type == 'all' and 'token' in self.temporal_fusion_type:
            # masking token randomly for temporal fusion during training
            if 'mask' in self.temporal_fusion_type and self.batch_size > 1:                
                if 'every' in self.temporal_fusion_type:
                    mask_interval = int([i for i in self.temporal_fusion_type.split('_') if 'every' in i][0].split('every')[-1])
                    # mask_indice = [i for i in range(1, self.frame_num, mask_interval)]
                    mask_indice_start = torch.randint(0, mask_interval, (1,))
                    mask_indice = [i for i in range(mask_indice_start, self.frame_num, mask_interval)]
                    mask = torch.ones(self.frame_num, device=head_feature.device)
                    mask[mask_indice] = 0
                    mask = mask[None, :].expand(self.batch_size, self.frame_num)
                elif 'mid' in self.temporal_fusion_type:
                    # mask_mid_pad = int([i for i in self.temporal_fusion_type.split('_') if 'mid' in i][0].split('mid')[-1])
                    # mask_mid_index = int(self.frame_num // 2)
                    # mask_indice = [i for i in range(mask_mid_index-mask_mid_pad, mask_mid_index+mask_mid_pad+1)]
                    mask_len = int([i for i in self.temporal_fusion_type.split('_') if 'mid' in i][0].split('mid')[-1])
                    mask_start_index = torch.randint(0, self.frame_num-mask_len, (1,))
                    mask_indice = [i for i in range(mask_start_index, mask_start_index+mask_len)]
                    mask = torch.ones(self.frame_num, device=head_feature.device)
                    mask[mask_indice] = 0
                    mask = mask[None, :].expand(self.batch_size, self.frame_num)
                elif 'random' in self.temporal_fusion_type:
                    mask_ratio = float([i for i in self.temporal_fusion_type.split('_') if 'random' in i][0].split('random')[-1])
                    mask_ratio = mask_ratio / 100
                    mask = torch.rand(self.batch_size, self.frame_num, device=head_feature.device)
                    mask = (mask < mask_ratio).float()
                else:
                    mask = torch.rand(self.batch_size, self.frame_num, device=head_feature.device)
                    mask = (mask < 0.5).float()
                
                ja_embedding_relation = ja_embedding_relation * mask[:, :, None]

            ja_token_tem_idx = torch.arange(self.frame_num).to(device=head_feature.device).long()
            ja_token_tem_idx = ja_token_tem_idx.view(1, self.frame_num).expand(self.batch_size, self.frame_num)
            ja_token_tem_enc = self.ja_token_tem_enc.to(device=head_feature.device)
            ja_token_tem_enc = ja_token_tem_enc[ja_token_tem_idx, :].view(self.batch_size, self.frame_num, self.people_feat_dim)
            ja_embedding_relation = ja_embedding_relation + ja_token_tem_enc
            for i in range(self.people_people_trans_enc_num):
                ja_embed_feat, ja_embedding_trans_weights = self.ja_embedding_self_attention[i](ja_embedding_relation, ja_embedding_relation, ja_embedding_relation)
                ja_embed_feat_res = ja_embedding_relation + ja_embed_feat
                ja_embed_feat_feed = self.ja_embedding_fc[i](ja_embed_feat_res)
                ja_embed_feat_feed_res = ja_embed_feat_res + ja_embed_feat_feed
                ja_embed_feat_feed_res = self.trans_layer_norm_ja_embedding(ja_embed_feat_feed_res)
                ja_embedding_relation = ja_embed_feat_feed_res
            # print(ja_embedding_trans_weights[0])
            # print(ja_embedding_trans_weights[0].argmax(dim=-1))

        if self.p_p_aggregation_type != 'token_only_concat':
            if 'fc' in self.p_p_estimator_type:
                ja_embedding_relation_input = ja_embedding_relation.view(self.batch_size, self.frame_num, 1, self.people_feat_dim)
            elif 'deconv' in self.p_p_estimator_type:
                ja_embedding_relation_input = ja_embedding_relation.view(self.batch_size, self.frame_num, 1, self.people_feat_dim)
                ja_embedding_relation_input = self.person_person_attention_heatmap_pre_fc(ja_embedding_relation_input)
                ja_embedding_relation_input = ja_embedding_relation_input.view(self.batch_size, self.frame_num, self.latent_ch, self.latent_height, self.latent_width)
            elif 'field' in self.p_p_estimator_type:
                ja_embedding_relation_view = ja_embedding_relation.view(self.batch_size, self.frame_num, 1, 1, self.people_feat_dim)
                ja_embedding_relation_expand = ja_embedding_relation_view.expand(self.batch_size, self.frame_num, 1, self.hm_height_middle*self.hm_width_middle, self.people_feat_dim)
                x_axis_map = xy_axis_map[:, 0, 0, :, :][:, None, :, :]
                y_axis_map = xy_axis_map[:, 0, 1, :, :][:, None, :, :]
                x_axis_map = F.interpolate(x_axis_map, (self.hm_height_middle, self.hm_width_middle), mode='bilinear')
                y_axis_map = F.interpolate(y_axis_map, (self.hm_height_middle, self.hm_width_middle), mode='bilinear')
                x_axis_map = x_axis_map.view(self.batch_size, 1, 1, self.hm_height_middle*self.hm_width_middle, 1)
                x_axis_map = x_axis_map.expand(self.batch_size, self.frame_num, 1, self.hm_height_middle*self.hm_width_middle, 1)
                y_axis_map = y_axis_map.view(self.batch_size, 1, 1, self.hm_height_middle*self.hm_width_middle, 1)
                y_axis_map = y_axis_map.expand(self.batch_size, self.frame_num, 1, self.hm_height_middle*self.hm_width_middle, 1)
                ja_embedding_relation_input = torch.cat([ja_embedding_relation_expand, x_axis_map, y_axis_map], dim=-1)
            person_person_joint_attention_heatmap = self.person_person_attention_heatmap(ja_embedding_relation_input)
            person_person_joint_attention_heatmap = person_person_joint_attention_heatmap.view(self.batch_size*self.frame_num, 1, self.hm_height_middle, self.hm_width_middle)
            person_person_joint_attention_heatmap = F.interpolate(person_person_joint_attention_heatmap, (self.hm_height, self.hm_width), mode='bilinear')
        else:
            ja_embedding_relation_view = head_info_params_emb[:, :, :-1, :].view(self.batch_size, self.frame_num, 1, 1, -1)
            ja_embedding_relation_expand = ja_embedding_relation_view.expand(self.batch_size, self.frame_num, 1, self.hm_height_middle*self.hm_width_middle, self.people_feat_dim*12)
            x_axis_map = xy_axis_map[:, 0, 0, :, :][:, None, :, :]
            y_axis_map = xy_axis_map[:, 0, 1, :, :][:, None, :, :]
            x_axis_map = F.interpolate(x_axis_map, (self.hm_height_middle, self.hm_width_middle), mode='bilinear')
            y_axis_map = F.interpolate(y_axis_map, (self.hm_height_middle, self.hm_width_middle), mode='bilinear')
            x_axis_map = x_axis_map.view(self.batch_size, 1, 1, self.hm_height_middle*self.hm_width_middle, 1)
            x_axis_map = x_axis_map.expand(self.batch_size, self.frame_num, 1, self.hm_height_middle*self.hm_width_middle, 1)
            y_axis_map = y_axis_map.view(self.batch_size, 1, 1, self.hm_height_middle*self.hm_width_middle, 1)
            y_axis_map = y_axis_map.expand(self.batch_size, self.frame_num, 1, self.hm_height_middle*self.hm_width_middle, 1)
            ja_embedding_relation_input = torch.cat([ja_embedding_relation_expand, x_axis_map, y_axis_map], dim=-1)
            person_person_joint_attention_heatmap = self.person_person_attention_heatmap_concat(ja_embedding_relation_input)
            person_person_joint_attention_heatmap = person_person_joint_attention_heatmap.view(self.batch_size*self.frame_num, 1, self.hm_height_middle, self.hm_width_middle)
            person_person_joint_attention_heatmap = F.interpolate(person_person_joint_attention_heatmap, (self.hm_height, self.hm_width), mode='bilinear')
        person_person_joint_attention_heatmap = person_person_joint_attention_heatmap.view(self.batch_size, self.frame_num, 1, self.hm_height, self.hm_width)

        # final p_p heatmap aggregation
        no_pad_idx = torch.sum(head_feature, dim=-1) != 0
        if self.p_p_aggregation_type == 'ind_only':
            person_person_attention_heatmap = person_person_attention_heatmap * no_pad_idx[:, :, :, None, None]
            person_person_joint_attention_heatmap = torch.sum(person_person_attention_heatmap, dim=2)
            person_person_joint_attention_heatmap = person_person_joint_attention_heatmap / torch.sum(no_pad_idx, dim=2)[:, :, None, None]
            person_person_joint_attention_heatmap = person_person_joint_attention_heatmap[:, :, None, :, :]
        elif self.p_p_aggregation_type == 'token_only':
            pass
        elif self.p_p_aggregation_type == 'ind_and_token_ind_based':
            person_person_attention_heatmap = person_person_attention_heatmap * no_pad_idx[:, :, :, None, None]
            person_person_joint_attention_heatmap = torch.sum(person_person_attention_heatmap, dim=2)
            person_person_joint_attention_heatmap = person_person_joint_attention_heatmap / torch.sum(no_pad_idx, dim=2)[:, :, None, None]
            person_person_joint_attention_heatmap = person_person_joint_attention_heatmap[:, :, None, :, :]
        elif self.p_p_aggregation_type == 'ind_and_token_token_based':
            pass
        else:
            pass

        # attention estimation of person-to-scene path
        # person_scene_attention_heatmap = inp['person_scene_attention_heatmap']
        # person_scene_attention_heatmap = person_scene_attention_heatmap.view(self.batch_size, self.people_num, self.hm_height_middle_p_s, self.hm_width_middle_p_s)
        # person_scene_attention_heatmap = F.interpolate(person_scene_attention_heatmap, (self.hm_height_p_s, self.hm_width_p_s), mode='bilinear')
        # person_scene_attention_heatmap = person_scene_attention_heatmap * (torch.sum(head_feature, dim=2) != 0)[:, :, None, None]

        # joint attention estimation of person-to-scene path
        # person_scene_joint_attention_heatmap = torch.sum(person_scene_attention_heatmap, dim=1)
        # no_pad_idx_cnt = torch.sum((torch.sum(head_feature, dim=2) != 0), dim=1)
        # person_scene_joint_attention_heatmap = person_scene_joint_attention_heatmap / no_pad_idx_cnt[:, None, None]
        # person_scene_joint_attention_heatmap = person_scene_joint_attention_heatmap[:, None, :, :]

        # generate head xy map
        # print('head_xy_map', head_xy_map.shape)
        # head_xy_map = head_xy_map.unsqueeze(1)
        # print('head_xy_map', head_xy_map.shape)
        # head_xy_map = head_xy_map.expand(self.batch_size, self.frame_num, self.people_num, 2, self.hm_height, self.hm_width)
        # head_xy_map = head_xy_map * head_feature[:, :, :, :2, None, None]

        # generate gaze xy map
        # gaze_xy_map = gaze_xy_map.unsqueeze(1)
        # gaze_xy_map = gaze_xy_map.expand(self.batch_size, self.frame_num, self.people_num, 2, self.hm_height, self.hm_width)
        # gaze_xy_map = gaze_xy_map * head_vector[:, :, :, :2, None, None]

        # expand xy axis map
        # xy_axis_map = xy_axis_map.unsqueeze(1)
        # xy_axis_map = xy_axis_map.expand(self.batch_size, self.frame_num, self.people_num, 2, self.hm_height, self.hm_width)

        # generate gaze cone map
        # xy_axis_map_dif_head = xy_axis_map - head_xy_map
        # x_axis_map_dif_head_mul_gaze = xy_axis_map_dif_head * gaze_xy_map
        # xy_dot_product = torch.sum(x_axis_map_dif_head_mul_gaze, dim=-3)
        # xy_dot_product = xy_dot_product / (torch.norm(xy_axis_map_dif_head, dim=-3) + self.epsilon)
        # xy_dot_product = xy_dot_product / (torch.norm(gaze_xy_map, dim=-3) + self.epsilon)

        # calculate theta and distance map
        # theta_x_y = torch.acos(torch.clamp(xy_dot_product, -1+self.epsilon, 1-self.epsilon))
        # distance_x_y = torch.norm(xy_axis_map_dif_head, dim=-3)

        # normalize theta and distance
        # theta_x_y = theta_x_y / self.pi
        # distance_x_y = distance_x_y / (2**0.5)

        # angle_dist = torch.exp(-torch.pow(theta_x_y, 2)/(2* 0.5 ** 2))
        # angle_dist = angle_dist * (torch.sum(head_feature, dim=-1) != 0)[:, :, :, None, None]
        # distance_dist = torch.exp(-torch.pow(theta_x_y, 2)/(2* 0.5 ** 2))
        # distance_dist = distance_dist * (torch.sum(head_feature, dim=-1) != 0)[:, :, :, None, None]
        trans_att_people_rgb = torch.zeros(self.batch_size, self.frame_num, self.people_num, 1, self.hm_height, self.hm_height)

        # pack return values
        data = {}
        data['person_person_attention_heatmap'] = person_person_attention_heatmap
        data['person_person_joint_attention_heatmap'] = person_person_joint_attention_heatmap
        data['person_scene_attention_heatmap'] = person_person_attention_heatmap
        data['person_scene_joint_attention_heatmap'] = person_person_joint_attention_heatmap
        data['final_joint_attention_heatmap'] = person_person_joint_attention_heatmap

        # data['angle_dist'] = angle_dist
        # data['distance_dist'] = distance_dist
        data['trans_att_people_rgb'] = trans_att_people_rgb
        data['trans_att_people_people'] = trans_att_people_people
        data['head_info_params'] = head_info_params

        if self.use_attribute_loss_type != 'original':
            if self.use_action_loss:
                data['action_person_predicted'] = action_person_predicted
                if self.use_frame_type == 'all' and 'atb' in self.temporal_fusion_type:
                    data['action_person_predicted_update'] = action_person_predicted_update
            if self.use_gaze_loss:
                data['gaze_predicted'] = gaze_person_predicted
                if self.use_frame_type == 'all' and 'atb' in self.temporal_fusion_type:
                    data['gaze_person_predicted_update'] = gaze_person_predicted_update

        return data

    def calc_loss(self, inp, out, cfg):
        # unpack data (input)
        img_gt_attention = inp['img_gt']
        gt_box = inp['gt_box']
        gt_box_id = inp['gt_box_id']
        att_inside_flag = inp['att_inside_flag']
        head_feature = inp['head_feature']
        head_vector = inp['head_vector']

        # get variables
        self.batch_size, self.frame_num, self.people_num, _ = head_feature.shape

        # pack loss
        loss_set = {}

        # unpack data (output)
        person_person_attention_heatmap = out['person_person_attention_heatmap']
        person_person_joint_attention_heatmap = out['person_person_joint_attention_heatmap']
        self.use_person_person_att_loss = cfg.exp_params.use_person_person_att_loss
        self.use_person_person_jo_att_loss = cfg.exp_params.use_person_person_jo_att_loss

        # switch loss coeficient
        if self.use_person_person_att_loss:
            loss_person_person_att_coef = 1
        else:
            loss_person_person_att_coef = 0

        if self.use_person_person_jo_att_loss:
            use_person_person_jo_att_coef = 1
        else:
            use_person_person_jo_att_coef = 0

        # generate gt map
        img_gt_joint_attention = torch.sum(img_gt_attention, dim=2)
        img_gt_all_thresh = torch.ones(1, device=img_gt_attention.device)
        img_gt_joint_attention = torch.where(img_gt_joint_attention>img_gt_all_thresh, img_gt_all_thresh, img_gt_joint_attention)

        # calculate person-person path loss
        loss_p_p_att = self.loss_func_hm_sum(person_person_attention_heatmap.float(), img_gt_attention.float())
        loss_p_p_att = loss_p_p_att/(torch.sum(att_inside_flag)*self.resize_height*self.resize_width)
        loss_p_p_att = loss_person_person_att_coef * loss_p_p_att
        person_person_joint_attention_heatmap = person_person_joint_attention_heatmap[:, :, 0, :, :]
        loss_p_p_jo_att = self.loss_func_hm_mean(person_person_joint_attention_heatmap.float(), img_gt_joint_attention.float())
        loss_p_p_jo_att = use_person_person_jo_att_coef * loss_p_p_jo_att
        loss_set['loss_p_p_att'] = loss_p_p_att
        loss_set['loss_p_p_jo_att'] = loss_p_p_jo_att

        if self.use_attribute_loss_type != 'original':
            if self.use_action_loss:
                people_bbox_tensor = inp['people_bbox']
                people_bbox_flag = torch.sum(people_bbox_tensor, dim=-1) != 0
                action_person = head_feature[:, :, :, 2:].argmax(dim=-1).long().view(-1)
                action_person = action_person[people_bbox_flag.view(-1)]
                action_person_predicted = out['action_person_predicted'].reshape(self.batch_size*self.frame_num*self.people_num, self.use_action_class_num)
                action_person_predicted = action_person_predicted[people_bbox_flag.view(-1)]
                action_loss = F.cross_entropy(action_person_predicted, action_person, weight=self.action_loss_weight)
                loss_set['action_loss'] = action_loss * self.action_loss_coef
                if self.use_frame_type == 'all' and 'atb' in self.temporal_fusion_type:
                    action_person_predicted_update = out['action_person_predicted_update'].reshape(self.batch_size*self.frame_num*self.people_num, self.use_action_class_num)
                    action_person_predicted_update = action_person_predicted_update[people_bbox_flag.view(-1)]
                    action_loss_update = F.cross_entropy(action_person_predicted_update, action_person, weight=self.action_loss_weight)
                    loss_set['action_loss_update'] = action_loss_update * self.action_loss_coef
            
            if self.use_gaze_loss:
                people_bbox_tensor = inp['people_bbox']
                people_bbox_flag = torch.sum(people_bbox_tensor, dim=-1) != 0
                gaze_person = head_vector.view(-1, 2)
                gaze_person = gaze_person[people_bbox_flag.view(-1)]
                gaze_person_predicted = out['gaze_predicted'].view(-1, 2)
                gaze_person_predicted = gaze_person_predicted[people_bbox_flag.view(-1)]
                gaze_loss = F.mse_loss(gaze_person_predicted, gaze_person)
                loss_set['gaze_loss'] = gaze_loss * self.gaze_loss_coef
                if self.use_frame_type == 'all' and 'atb' in self.temporal_fusion_type:
                    gaze_person_predicted_update = out['gaze_person_predicted_update'].view(-1, 2)
                    gaze_person_predicted_update = gaze_person_predicted_update[people_bbox_flag.view(-1)]
                    gaze_loss_update = F.mse_loss(gaze_person_predicted_update, gaze_person)
                    loss_set['gaze_loss_update'] = gaze_loss_update * self.gaze_loss_coef

        return loss_set