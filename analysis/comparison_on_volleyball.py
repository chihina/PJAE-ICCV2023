import json
import os
import sys
import pandas as pd
import numpy as np

saved_result_dir = os.path.join('results', 'volleyball')

# define analyze model type
analyze_name_list_dic = {}

analyze_name_list = []
analyze_name_list.append('2021_0708_lr_e3_gamma_1_stack_3_mid_frame_ver2')
analyze_name_list.append('volleyball-isa_bbox_GT_gaze_GT_act_GT')
analyze_name_list.append('volleyball-isa_bbox_PRED_gaze_PRED_act_PRED')
analyze_name_list.append('volleyball-dual-mid_p_p_field_middle_p_s_davt_bbox_GT_gaze_GT_act_GT_psfix_fusion_wo_p_p')
analyze_name_list.append('volleyball-dual-mid_p_p_field_middle_p_s_davt_bbox_PRED_gaze_PRED_act_PRED_psfix_fusion_wo_p_p')
analyze_name_list.append('volleyball-dual-mid_p_p_field_middle_p_s_davt_bbox_GT_gaze_GT_act_GT_weight_fusion_fine_token_only')
analyze_name_list.append('volleyball-dual-mid_p_p_field_middle_p_s_davt_bbox_PRED_gaze_PRED_act_PRED_weight_fusion_fine_token_only')
analyze_name_list_dic['bbox_GT_gaze_GT_act_GT_blur_False'] = analyze_name_list

# analyze_name_list = []
# analyze_name_list.append('2021_0708_lr_e3_gamma_1_stack_3_mid_frame_ver2')
# analyze_name_list.append('volleyball-isa_bbox_GT_gaze_GT_act_GT')
# analyze_name_list.append('volleyball-isa_bbox_PRED_gaze_PRED_act_PRED')
# analyze_name_list.append('volleyball-dual-mid_p_p_field_middle_p_s_davt_bbox_GT_gaze_GT_act_GT_psfix_fusion_wo_p_p')
# analyze_name_list.append('volleyball-dual-mid_p_p_field_middle_p_s_davt_bbox_PRED_gaze_PRED_act_PRED_psfix_fusion_wo_p_p')
# analyze_name_list.append('volleyball-dual-mid_p_p_field_middle_p_s_davt_bbox_GT_gaze_GT_act_GT_psfix_fusion_scalar_weight_fine')
# analyze_name_list.append('volleyball-dual-mid_p_p_field_middle_p_s_davt_bbox_PRED_gaze_PRED_act_PRED_weight_fusion_fine_token_only')
# analyze_name_list_dic['bbox_PRED_gaze_PRED_act_PRED_blur_False'] = analyze_name_list

# define model names
model_name_list = []
model_name_list.append('Ball detection')
model_name_list.append('ISA (Trained on ground-truth attributes)')
model_name_list.append('ISA (Trained on predicted attributes)')
model_name_list.append('DAVT (Trained on grond-truth attributes)')
model_name_list.append('DAVT (Trained on predicted attributes)')
model_name_list.append('Ours (Trained on grond-truth attributes)')
model_name_list.append('Ours (Trained on predicted attributes)')

for test_data_type, analyze_name_list in analyze_name_list_dic.items():
    print(f'==={test_data_type}===')
    eval_results_list = []
    for analyze_name in analyze_name_list:
        analyze_name_type = analyze_name
        model_name = f'{analyze_name_type}'        
        json_file_path = os.path.join(saved_result_dir, model_name, 'eval_results', test_data_type, 'eval_results.json')
        with open(json_file_path, 'r') as f:
            eval_results_dic = json.load(f)
        eval_results_list.append(list(eval_results_dic.values()))
        eval_metrics_list = list(eval_results_dic.keys())

    eval_results_array = np.array(eval_results_list)
    df_eval_results = pd.DataFrame(eval_results_array, model_name_list, eval_metrics_list)
    save_csv_file_path = os.path.join(saved_result_dir, f'comparision_volleyball_{test_data_type}.csv')
    df_eval_results.to_csv(save_csv_file_path)