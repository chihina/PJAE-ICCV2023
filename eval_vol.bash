IFS_BACKUP=$IFS
IFS=$'\n'

model_array=(
    # Please fill in the model path
    # 'volleyball_GT_ori_att_vid_token_t_enc'
    # 'volleyball_GT_ori_att_vid_token_mask_t_enc'
    # 'volleyball_PRED_ori_att_vid_token_t_enc'
    # 'volleyball_PRED_ori_att_vid_token_mask_t_enc'
    # 'volleyball_PRED_ori_att_vid_wo_token'
    # 'volleyball_PRED_ori_att_vid_token_mask_every2_t_enc'
    # 'volleyball_PRED_ori_att_vid_token_mask_mid1_t_enc'
    # 'volleyball_PRED_ori_att_vid_token_mask_random10_t_enc'
    # 'volleyball_PRED_ori_att_vid_token_mask_random25_t_enc'
    # 'volleyball_PRED_ori_att_vid_token_mask_random75_t_enc'
    # 'volleyball_PRED_ori_att_vid_token_mask_random90_t_enc'
    # 'volleyball_PRED_ori_att_vid_token_mask_random25_t_enc_DAVT_scalar_fusion'
    # 'volleyball_PRED_ori_att_vid_token_mask_random25_t_enc_DAVT_scalar_fusion_freeze'
    # 'volleyball_PRED_DAVT_only'
    # 'volleyball_PRED_DAVT_only_lr_e2_modified'
    # 'volleyball_PRED_DAVT_only_lr_e3_modified'
    # 'volleyball_PRED_DAVT_only_lr_e3_modified_init'
    # 'volleyball_PRED_DAVT_only_lr_e3_modified_init_2layer'
    # 'volleyball_PRED_DAVT_only_lr_e3_modified_videoatttarget'
    # 'volleyball_PRED_ori_att_vid_token_mask_random25_t_enc_DAVT_scalar_fusion_mod'
    # 'volleyball_PRED_DAVT_only_lr_e2'
    'volleyball_PRED_DAVT_only_lr_e3_demo'
    'volleyball_PRED_DAVT_only_lr_e3_gazefollow'
    # 'volleyball_PRED_DAVT_only_lr_e3_videoatttarget'
    # 'volleyball_PRED_DAVT_only_lr_e3'
    # 'volleyball_PRED_DAVT_only_lr_e4'
    # 'volleyball_PRED_ori_att_vid_token_mask_random25_t_enc_DAVT_scalar_fusion_ver2'
    # 'volleyball_PRED_ori_att_vid_token_mask_random25_t_enc_DAVT_scalar_fusion_ver2_freeze'
    # 'volleyball_PRED_ori_att_vid_token_mask_every2_start_random_t_enc'
    # 'volleyball_PRED_ori_att_vid_token_mask_mid3_start_random_t_enc'
    # 'volleyball_PRED_ori_att_vid_token_mask_mid4_start_random_t_enc'
    # 'volleyball_PRED_ori_att_vid_token_mask_random25_t_enc_DAVT_scalar_fusion'
    )

for model in ${model_array[@]}; do
  echo $model
  python eval_on_volleyball_ours.py yaml_files/volleyball_wo_att/eval.yaml -m $model
done