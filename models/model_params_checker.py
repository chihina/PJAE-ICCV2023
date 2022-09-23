import torch
import torch.nn as nn
from torch.nn import functional as F
import sys
import  numpy as np

class PersonToPersonEstimator(nn.Module):
    def __init__(self, p_p_estimator_type):
        super(PersonToPersonEstimator, self).__init__()

        self.p_p_estimator_type = p_p_estimator_type
        self.resize_height = 320
        self.resize_width = 480
        self.people_feat_dim = 16

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
                nn.ConvTranspose2d(self.latent_ch, self.latent_ch//2, 4, 2, 1, bias=False),
                nn.ReLU(),
                nn.ConvTranspose2d(self.latent_ch//2, self.latent_ch//2, 4, 2, 1, bias=False),
                nn.ReLU(),
                nn.ConvTranspose2d(self.latent_ch//2, self.latent_ch//4, 4, 2, 1, bias=False),
                nn.ReLU(),
                nn.ConvTranspose2d(self.latent_ch//4, self.latent_ch//8, 4, 2, 1, bias=False),
                nn.ReLU(),
                nn.ConvTranspose2d(self.latent_ch//8, 1, 4, 2, 1, bias=False),
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
            )
        else:
            print('please use correct p_p estimator type')
            sys.exit()

    def forward(self, inp):
        data = {}
        return data

p_p_estimator_type_list = ['field_shallow', 'field_middle', 'field_deep', 
                           'fc_shallow', 'fc_middle', 'fc_deep',
                           'deconv_shallow', 'deconv_middle', 'deconv_deep',
                           ]

for p_p_estimator_type in p_p_estimator_type_list:
    print(p_p_estimator_type)
    model = PersonToPersonEstimator(p_p_estimator_type)
    params = 0
    for p in model.parameters():
        if p.requires_grad:
            params += p.numel()
    print(params)