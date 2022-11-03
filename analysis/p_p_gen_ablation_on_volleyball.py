import json
import os
import sys
import pandas as pd
import numpy as np


saved_result_dir = os.path.join('results', 'volleyball')

# define analyze model type
analyze_name_list = []
analyze_name_list.append('volleyball-dual-mid_p_p_field_middle_bbox_GT_gaze_GT_act_GT')
analyze_name_list.append('volleyball-dual-mid_p_p_field_middle_bbox_PRED_gaze_PRED_act_PRED')

# define ablate type
analyze_name_ablation_list = []
analyze_name_ablation_list.append('fc_shallow')
analyze_name_ablation_list.append('fc_middle')
analyze_name_ablation_list.append('fc_deep')
analyze_name_ablation_list.append('deconv_shallow')
analyze_name_ablation_list.append('deconv_middle')
analyze_name_ablation_list.append('field_middle')
analyze_name_ablation_list.append('field_deep')

# define test data type
test_data_type_list = []
test_data_type_list.append('bbox_GT_gaze_GT_act_GT_blur_False')
test_data_type_list.append('bbox_PRED_gaze_PRED_act_PRED_blur_False')
for test_data_type in test_data_type_list:
    print(f'==={test_data_type}===')
    for analyze_name in analyze_name_list:
        model_name_list = []
        eval_results_list = []
        analyze_name_type = analyze_name.split('_')[0]
        for ablation_name in analyze_name_ablation_list:

            model_name = f'{analyze_name_type}_{ablation_name}'        
            model_name_list.append(model_name)

            json_file_path = os.path.join(saved_result_dir, model_name, 'eval_results', test_data_type, 'eval_results.json')
            with open(json_file_path, 'r') as f:
                eval_results_dic = json.load(f)
            eval_results_list.append(list(eval_results_dic.values()))
            eval_metrics_list = list(eval_results_dic.keys())

        eval_results_array = np.array(eval_results_list)
        df_eval_results = pd.DataFrame(eval_results_array, model_name_list, eval_metrics_list)
        save_csv_file_path = os.path.join(saved_result_dir, f'p_p_gen_ablation_{test_data_type}.csv')
        df_eval_results.to_csv(save_csv_file_path)