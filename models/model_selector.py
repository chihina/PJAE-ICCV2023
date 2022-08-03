from models.head_pose_estimator import HeadPoseEstimatorResnet
from models.joint_attention_estimator import JointAttentionEstimatorTransformer

def model_generator(cfg):
    if cfg.model_params.model_type == 'ja_transformer':
        model_head = HeadPoseEstimatorResnet(cfg)
        model_gaussian = JointAttentionEstimatorTransformer(cfg)
    else:
        assert True, 'cfg.exp_parames.model_type is incorrect'

    return model_head, model_gaussian, cfg