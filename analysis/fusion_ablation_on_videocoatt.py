import json
import os
import sys
import pandas as pd
import numpy as np


saved_result_dir = os.path.join('results', 'videocoatt')

# define analyze model type
analyze_name_list = []
analyze_name_list.append('videocoatt-p_p_field_deep_p_s')

# define ablate type
analyze_name_ablation_list = []
analyze_name_ablation_list.append('davt_simple_average')
analyze_name_ablation_list.append('davt_scalar_weight')
analyze_name_ablation_list.append('davt_freeze')

# define model names
model_name_list = []
model_name_list.append('Mean average')
model_name_list.append('Weighted average')
model_name_list.append('CNN fusion')

# define test data type
test_data_type_list = []
test_data_type_list.append('test_gt_gaze_False_head_conf_0.6')
for test_data_type in test_data_type_list:
    print(f'==={test_data_type}===')
    for analyze_name in analyze_name_list:
        eval_results_list = []
        for ablation_name in analyze_name_ablation_list:

            model_name = f'{analyze_name}{ablation_name}'        

            json_file_path = os.path.join(saved_result_dir, model_name, 'eval_results', test_data_type, 'eval_results.json')
            with open(json_file_path, 'r') as f:
                eval_results_dic = json.load(f)
            eval_results_list.append(list(eval_results_dic.values()))
            eval_metrics_list = list(eval_results_dic.keys())

        eval_results_array = np.array(eval_results_list)
        df_eval_results = pd.DataFrame(eval_results_array, model_name_list, eval_metrics_list)
        save_csv_file_path = os.path.join(saved_result_dir, f'fusion_ablation_videocoatt_{test_data_type}.csv')
        df_eval_results.to_csv(save_csv_file_path)