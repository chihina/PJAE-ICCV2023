import json
import os
import sys
import pandas as pd
import numpy as np

sys.path.append('./')
from analysis.utils import refine_excel

saved_result_dir = os.path.join('results', 'volleyball_wo_att')

# define ablate type
analyze_name_list = []
analyze_name_list.append('volleyball-isa_bbox_PRED_gaze_PRED_action_PRED_vid')
analyze_name_list.append('volleyball_PRED_DAVT_only_lr_e3')
analyze_name_list.append('volleyball_PRED_DAVT_only_lr_e3_demo')
analyze_name_list.append('volleyball_PRED_DAVT_only_lr_e3_gazefollow')
analyze_name_list.append('volleyball_PRED_DAVT_only_lr_e3_videoatttarget')
analyze_name_list.append('volleyball_PRED_ori_att_vid_token_mask_random25_t_enc_DAVT_scalar_fusion_mod')

# define model name
model_name_list = []
model_name_list.append('ISA')
model_name_list.append('DAVT (init)')
model_name_list.append('DAVT (demo)')
model_name_list.append('DAVT (gaze)')
model_name_list.append('DAVT (videoatt)')
model_name_list.append('Ours')

# define test data type
test_data_type_list = []
# test_data_type_list.append('bbox_GT_gaze_GT_act_GT_blur_False')
test_data_type_list.append('bbox_PRED_gaze_PRED_act_PRED_blur_False')
for test_data_type in test_data_type_list:
    print(f'==={test_data_type}===')
    eval_results_list = []
    for analyze_name in analyze_name_list:
        model_name = f'{analyze_name}'        

        json_file_path = os.path.join(saved_result_dir, model_name, 'eval_results', test_data_type, 'eval_results.json')

        with open(json_file_path, 'r') as f:
            eval_results_dic = json.load(f)
        eval_results_list.append(list(eval_results_dic.values()))
        eval_metrics_list = list(eval_results_dic.keys())

    eval_results_array = np.array(eval_results_list)
    df_eval_results = pd.DataFrame(eval_results_array, model_name_list, eval_metrics_list)
    save_excel_file_path = os.path.join(saved_result_dir, f'comparison_on_volleyball_{test_data_type}.xlsx')
    df_eval_results.to_excel(save_excel_file_path, sheet_name='all')
    refine_excel(save_excel_file_path)