from models.head_pose_estimator import HeadPoseEstimatorResnet
from models.joint_attention_estimator_transformer import JointAttentionEstimatorTransformer
from models.inferring_shared_attention_estimation import InferringSharedAttentionEstimator

import sys

def model_generator(cfg):
    if cfg.model_params.model_type == 'ja_transformer':
        model_head = HeadPoseEstimatorResnet(cfg)
        model_gaussian = JointAttentionEstimatorTransformer(cfg)
    elif cfg.model_params.model_type == 'isa':
        model_head = HeadPoseEstimatorResnet(cfg)
        model_gaussian = InferringSharedAttentionEstimator(cfg)
    else:
        assert True, 'cfg.exp_parames.model_type is incorrect'

    return model_head, model_gaussian, cfg