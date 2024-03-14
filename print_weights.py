import os
import torch

from models.model_selector import model_generator
from models.davt_scene_extractor import ModelSpatial, ModelSpatialDummy, ModelSpatioTemporal


gpus_list = [4]
print("===> Load pretrained model (saliecny extractor)")
model_name = 'pretrained_scene_extractor_davt'
weight_name_list = ['model_demo.pt', 'model_gazefollow.pt', 'model_videoatttarget.pt', 'initial_weights_for_temporal_training.pt']
weight_key_dict = {}
for weight_name in weight_name_list:
    model_weight_path = os.path.join('saved_weights', 'volleyball_wo_att', model_name, weight_name)
    pretrained_dict = torch.load(model_weight_path,  map_location='cuda:'+str(gpus_list[0]))
    pretrained_dict = pretrained_dict['model']
    weight_key_len = len(pretrained_dict.keys())
    print(weight_name, weight_key_len)
    weight_key_dict[weight_name] = list(pretrained_dict.keys())

weight_img = weight_key_dict[weight_name_list[0]]
weight_vid = weight_key_dict[weight_name_list[2]]
weight_diff = sorted(set(weight_vid) - set(weight_img))
# print(weight_diff)
# for weihgt_diff_key in weight_diff:
    # print(weihgt_diff_key)

model_saliency = ModelSpatioTemporal()
model_saliency_dict = model_saliency.state_dict()
model_saliency_dict_key_len = len(model_saliency_dict.keys())
print("model_saliency_dict_key_len", model_saliency_dict_key_len)

weight_vid_saliency = model_saliency_dict.keys()
weight_diff = sorted(set(weight_vid_saliency) - set(weight_img))
# print(weight_diff)
for weihgt_diff_key in weight_diff:
    print(weihgt_diff_key)

# model_saliency_dict.update(pretrained_dict)
# model_saliency.load_state_dict(model_saliency_dict)