import json
import os
import sys
import pandas as pd
import numpy as np


saved_result_dir = os.path.join('results', 'volleyball')

# define analyze model type
analyze_name_list = []
analyze_name_list.append('volleyball-dual-all_ver6')

# define ablate type
analyze_name_ablation_list = []
analyze_name_ablation_list.append('')
analyze_name_ablation_list.append('_wo_action')
analyze_name_ablation_list.append('_wo_gaze')
analyze_name_ablation_list.append('_wo_position')
analyze_name_ablation_list.append('_wo_p_p')
analyze_name_ablation_list.append('_wo_p_s')

# define test data type
test_data_type_list = []
test_data_type_list.append('bbox_GT_gaze_GT_act_GT')
test_data_type_list.append('bbox_PRED_gaze_PRED_act_PRED')
for test_data_type in test_data_type_list:
    print(f'==={test_data_type}===')
    for analyze_name in analyze_name_list:
        model_name_list = []
        eval_results_list = []
        analyze_name_type, analyze_name_prefix = analyze_name.split('_')
        for ablation_name in analyze_name_ablation_list:

            model_name = f'{analyze_name_type}{ablation_name}_{analyze_name_prefix}'        
            model_name_list.append(model_name)

            json_file_path = os.path.join(saved_result_dir, model_name, 'eval_results', test_data_type, 'eval_results.json')
            with open(json_file_path, 'r') as f:
                eval_results_dic = json.load(f)
            eval_results_list.append(list(eval_results_dic.values()))
            eval_metrics_list = list(eval_results_dic.keys())

        eval_results_array = np.array(eval_results_list)
        df_eval_results = pd.DataFrame(eval_results_array, model_name_list, eval_metrics_list)
        save_csv_file_path = os.path.join(saved_result_dir, f'{analyze_name}_{test_data_type}.csv')
        df_eval_results.to_csv(save_csv_file_path)