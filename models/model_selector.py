from models.head_pose_estimator import HeadPoseEstimatorResnet
from models.joint_attention_estimator_transformer import JointAttentionEstimatorTransformer
from models.joint_attention_estimator_transformer_dual import JointAttentionEstimatorTransformerDual
from models.joint_attention_estimator_transformer_dual_only_people import JointAttentionEstimatorTransformerDualOnlyPeople
from models.joint_attention_estimator_transformer_dual_img_feat import JointAttentionEstimatorTransformerDualImgFeat
from models.joint_attention_estimator_transformer_dual_img_feat_only_people import JointAttentionEstimatorTransformerDualOnlyPeopleImgFeat
from models.joint_attention_fusion import JointAttentionFusion, JointAttentionFusionDummy
from models.inferring_shared_attention_estimation import InferringSharedAttentionEstimator
from models.end_to_end_human_gaze_target import EndToEndHumanGazeTargetTransformer
from models.davt_scene_extractor import ModelSpatial, ModelSpatialDummy, ModelSpatioTemporal
from models.transformer_scene_extractor import SceneFeatureTransformer
from models.cnn_scene_extractor import SceneFeatureCNN
from models.hourglass import HourglassNet
import sys

def model_generator(cfg):
    if cfg.model_params.model_type == 'ja_transformer':
        model_head = HeadPoseEstimatorResnet(cfg)
        model_gaussian = JointAttentionEstimatorTransformer(cfg)
        model_saliency = ModelSpatial()
    elif cfg.model_params.model_type == 'ja_transformer_dual':
        model_head = HeadPoseEstimatorResnet(cfg)
        model_gaussian = JointAttentionEstimatorTransformerDual(cfg)
        if cfg.model_params.p_s_estimator_type == 'davt':
            model_saliency = ModelSpatial()
        elif cfg.model_params.p_s_estimator_type == 'cnn':
            model_saliency = SceneFeatureCNN(cfg)
        elif cfg.model_params.p_s_estimator_type == 'transformer':
            model_saliency = SceneFeatureTransformer(cfg)
        model_fusion = JointAttentionFusion(cfg)
    elif cfg.model_params.model_type == 'ja_transformer_dual_only_people':
        model_head = HeadPoseEstimatorResnet(cfg)
        model_gaussian = JointAttentionEstimatorTransformerDualOnlyPeople(cfg)
        model_saliency = ModelSpatialDummy()
        model_fusion = JointAttentionFusionDummy()
    elif cfg.model_params.model_type == 'ja_transformer_dual_img_feat':
        model_head = HeadPoseEstimatorResnet(cfg)
        model_gaussian = JointAttentionEstimatorTransformerDualImgFeat(cfg)
        if cfg.model_params.p_s_estimator_type == 'davt':
            if cfg.exp_params.use_frame_type == 'mid':
                model_saliency = ModelSpatial()
            else:
                model_saliency = ModelSpatioTemporal(num_lstm_layers = 2)
        elif cfg.model_params.p_s_estimator_type == 'cnn':
            model_saliency = SceneFeatureCNN(cfg)
        elif cfg.model_params.p_s_estimator_type == 'transformer':
            model_saliency = SceneFeatureTransformer(cfg)
        model_fusion = JointAttentionFusion(cfg)
    elif cfg.model_params.model_type == 'ja_transformer_dual_only_people_img_feat':
        model_head = HeadPoseEstimatorResnet(cfg)
        model_gaussian = JointAttentionEstimatorTransformerDualOnlyPeopleImgFeat(cfg)
        model_saliency = ModelSpatialDummy()
        model_fusion = JointAttentionFusionDummy()
    elif cfg.model_params.model_type == 'isa':
        model_head = HeadPoseEstimatorResnet(cfg)
        model_gaussian = InferringSharedAttentionEstimator(cfg)
        if 'volleyball' in cfg.data.name:
            model_saliency = HourglassNet(3, 3, 5)
        else:
            model_saliency = ModelSpatialDummy(cfg)
        model_fusion = JointAttentionFusionDummy()
    elif cfg.model_params.model_type == 'human_gaze_target_transformer':
        model_head = HeadPoseEstimatorResnet(cfg)
        model_gaussian = EndToEndHumanGazeTargetTransformer(cfg)
        model_saliency = ModelSpatialDummy(cfg)
        model_fusion = JointAttentionFusionDummy()
    elif cfg.model_params.model_type == 'ball_detection':
        model_head = None
        model_gaussian = None
        model_saliency = HourglassNet(3, 3, 5)
        model_fusion = JointAttentionFusionDummy()
    else:
        assert True, 'cfg.exp_parames.model_type is incorrect'
    
    return model_head, model_gaussian, model_saliency, model_fusion, cfg