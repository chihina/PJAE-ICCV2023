from models.head_pose_estimator import HeadPoseEstimatorResnet
from models.joint_attention_estimator_transformer import JointAttentionEstimatorTransformer
from models.joint_attention_estimator_transformer_dual import JointAttentionEstimatorTransformerDual
from models.joint_attention_estimator_transformer_dual_only_people import JointAttentionEstimatorTransformerDualOnlyPeople
from models.inferring_shared_attention_estimation import InferringSharedAttentionEstimator
from models.end_to_end_human_gaze_target import EndToEndHumanGazeTargetTransformer
from models.davt_scene_extractor import ModelSpatial, ModelSpatialDummy

import sys

def model_generator(cfg):
    if cfg.model_params.model_type == 'ja_transformer':
        model_head = HeadPoseEstimatorResnet(cfg)
        model_gaussian = JointAttentionEstimatorTransformer(cfg)
        model_saliency = ModelSpatial()
    elif cfg.model_params.model_type == 'ja_transformer_dual':
        model_head = HeadPoseEstimatorResnet(cfg)
        model_gaussian = JointAttentionEstimatorTransformerDual(cfg)
        model_saliency = ModelSpatial()
    elif cfg.model_params.model_type == 'ja_transformer_only_people':
        model_head = HeadPoseEstimatorResnet(cfg)
        model_gaussian = JointAttentionEstimatorTransformerDualOnlyPeople(cfg)
        model_saliency = ModelSpatialDummy()
    elif cfg.model_params.model_type == 'isa':
        model_head = HeadPoseEstimatorResnet(cfg)
        model_gaussian = InferringSharedAttentionEstimator(cfg)
        model_saliency = ModelSpatialDummy(cfg)
    elif cfg.model_params.model_type == 'human_gaze_target_transformer':
        model_head = HeadPoseEstimatorResnet(cfg)
        model_gaussian = EndToEndHumanGazeTargetTransformer(cfg)
        model_saliency = ModelSpatialDummy(cfg)
    else:
        assert True, 'cfg.exp_parames.model_type is incorrect'
    
    return model_head, model_gaussian, model_saliency, cfg