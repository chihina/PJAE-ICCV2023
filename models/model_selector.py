from models.head_pose_estimator import HeadPoseEstimatorResnet
from models.joint_attention_estimator_transformer import JointAttentionEstimatorTransformer
from models.joint_attention_estimator_cnn import JointAttentionEstimatorCNN

def model_generator(cfg):
    if cfg.model_params.model_type == 'ja_transformer':
        model_head = HeadPoseEstimatorResnet(cfg)
        model_gaussian = JointAttentionEstimatorTransformer(cfg)
    elif cfg.model_params.model_type == 'ja_cnn':
        model_head = HeadPoseEstimatorResnet(cfg)
        model_gaussian = JointAttentionEstimatorCNN(cfg)
    else:
        assert True, 'cfg.exp_parames.model_type is incorrect'

    return model_head, model_gaussian, cfg