import json
import os
import sys
import pandas as pd
import numpy as np

saved_result_dir_analyze = os.path.join('results', 'volleyball_wo_att')

saved_result_dir_list = []
saved_result_dir_list.append(os.path.join('results', 'volleyball_all'))
saved_result_dir_list.append(os.path.join('results', 'volleyball_all'))
saved_result_dir_list.append(os.path.join('results', 'volleyball_wo_att'))

analyze_name_list = []
# analyze_name_list.append('volleyball-dual-mid_p_p_field_middle_p_s_davt_bbox_GT_gaze_GT_act_GT_weight_fusion_fine_token_only')
# analyze_name_list.append('volleyball-dual-mid_p_p_field_middle_p_s_davt_bbox_PRED_gaze_PRED_act_PRED_weight_fusion_fine_token_only')
analyze_name_list.append('volleyball-dual-mid_p_p_field_middle_bbox_GT_gaze_GT_act_GT')
analyze_name_list.append('volleyball-dual-mid_p_p_field_middle_bbox_PRED_gaze_PRED_act_PRED_token_only')
analyze_name_list.append('volleyball_p_p_field_middle_bbox_PRED_ind_128_token_only_w_gaze_loss_img_att_cross')

model_name_list = []
model_name_list.append('Ours (ICCV2023:GT)')
model_name_list.append('Ours (ICCV2023:PRED)')
model_name_list.append('Ours (PRED)')

# define test data type
test_data_type_list = []
# test_data_type_list.append('bbox_GT_gaze_GT_act_GT_blur_False')
test_data_type_list.append('bbox_PRED_gaze_PRED_act_PRED_blur_False')
for test_data_type in test_data_type_list:
    print(f'==={test_data_type}===')
    eval_results_list = []
    for analyze_idx, analyze_name in enumerate(analyze_name_list):
        model_name = f'{analyze_name}'        

        json_file_path = os.path.join(saved_result_dir_list[analyze_idx], model_name, 'eval_results', test_data_type, 'eval_results.json')

        with open(json_file_path, 'r') as f:
            eval_results_dic = json.load(f)
        eval_results_list.append(list(eval_results_dic.values()))
        eval_metrics_list = list(eval_results_dic.keys())

    eval_results_array = np.array(eval_results_list)
    df_eval_results = pd.DataFrame(eval_results_array, model_name_list, eval_metrics_list)
    save_excel_file_path = os.path.join(saved_result_dir_analyze, f'iccv_comparison_{test_data_type}.xlsx')
    df_eval_results.to_excel(save_excel_file_path)