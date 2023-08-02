# Intro

This is the official repository for the following paper:

Interaction-aware Joint Attention Estimation Using People Attributes, ICCV2023
(toyota-ti.ac.jp/Lab/Denshi/iim/ukita/selection/ICCV2023-PJAE.html)

## Environment
python 3.6.9

And you can use requirements.txt
```
pip install -r requirements.txt
```

# Data preparation
## 1. Download dataset
You can download daatset from the following url.  
These dataset are required to place in data/ in the repository.  

* Volleyball dataset (data/videos)  
https://github.com/mostafa-saad/deep-activity-rec

* Volleyball dataset (data/jae_dataset_bbox_gt, data/jae_dataset_bbox_pred)  
https://drive.google.com/drive/folders/1O55_wri92uv87g-2aDh8ll6dFVupmFaB?usp=share_link

* Volleyball dataset (data/vatic_ball_annotation/annotation_data)  
https://drive.google.com/drive/folders/1O55_wri92uv87g-2aDh8ll6dFVupmFaB?usp=share_link

* VideoCoAtt dataset (data/VideoCoAtt_Dataset)  
http://www.stat.ucla.edu/~lifengfan/shared_attention

* VideoCoAtt dataset (data/VideoCoAtt_Dataset/dets_heads)  
https://drive.google.com/drive/folders/1O55_wri92uv87g-2aDh8ll6dFVupmFaB?usp=share_link

## 2. Training
* You can change parameters of the model (e.g., multi-head numbers, transformer encoder numbers, ...) by editing the yaml files.
* Trained model are also published in here (https://drive.google.com/drive/folders/1O55_wri92uv87g-2aDh8ll6dFVupmFaB?usp=share_link
)
* trained models required to place in saved_weights/volleyball or saved_weights/videocoatt in the repository.


### 2.1 Volleyball dataset

* Ours
```
python train.py yaml/volleyball/train_ours_p_p.yaml
python train.py yaml/volleyball/train_ours.yaml
```
The following folder contains the trained models.
1. volleyball-dual-mid_p_p_field_middle_p_s_davt_bbox_PRED_gaze_PRED_act_PRED_weight_fusion_fine_token_only (Ex.1)
2. volleyball-dual-mid_p_p_field_middle_p_s_davt_bbox_GT_gaze_GT_act_GT_weight_fusion_fine_token_only (Ex.2)

* DAVT
```
python train.py yaml/volleyball/train_ours_p_p.yaml
python train.py yaml/volleyball/train_ours.yaml
```
The following folder contains the trained models.
1. volleyball-dual-mid_p_p_field_middle_p_s_davt_bbox_PRED_gaze_PRED_act_PRED_p_s_only (Ex.1)
2. volleyball-dual-volleyball-dual-mid_p_p_field_middle_p_s_davt_bbox_GT_gaze_GT_act_GT_p_s_only (Ex.2)

* ISA
```
python train.py yaml/volleyball/train_ours_isa.yaml
```
The following folder contains the trained models.
1. volleyball-dual-isa_bbox_PRED_gaze_PRED_act_PRED (Ex.1)
2. volleyball-dual-isa_bbox_GT_gaze_GT_act_GT (Ex.2)


### 2.2 VideoCoAtt dataset

* Ours
```
python train.py yaml/videocoatt/train_ours_p_p.yaml
python train.py yaml/videocoatt/train_ours.yaml
```
The following folder contains the trained models.
1. videocoatt-dual-p_p_field_deep_p_s_davt_scalar_weight_fix (Ex.1)
2. videocoatt-dual-p_p_field_deep_p_s_davt_scalar_weight_fix_token_only_GT (Ex.2)

* DAVT  
Trained model is published in here (https://github.com/ejcgt/attention-target-detection)

* ISA
```
python train.py yaml/videocoatt/train_ours_isa.yaml
```
The following folder contains the trained models.
1. videocoatt-isa_bbox_PRED_gaze_PRED (Ex.1)
2. videocoatt-isa_bbox_GT_gaze_GT (Ex.2)

* HGTD
```
python train.py yaml/videoattentiontarget/train_hgt.yaml
```
The following folder contains the trained models.
1. videocoatt-videoattentiontarget-hgt-high (Ex.1 and Ex.2)

## 3. Evaluation
### 3.1 Volleyball dataset
You can choose the model which you would like to evaluate in the yaml files.

* Ours and DAVT
```
python eval_on_volleyball_ours.py yaml/volleyball/eval.yaml
```

* ISA
```
python eval_on_videocoatt_isa.py yaml/volleyball/eval.yaml
```

### 3.2 VideoCoAtt dataset

* Ours and DAVT
```
python eval_on_videocoatt_ours.py yaml/videocoatt/eval.yaml
```

* ISA
```
python eval_on_videocoatt_isa.py yaml/videocoatt/eval.yaml
```

* HGTD
```
python eval_on_videocoatt_hgt.py yaml/videocoatt/eval.yaml
```

## 4. Demo
You can choose the model which you would like to evaluate in the yaml files.

### 4.1 Volleyball dataset
```
python demo_ours.py yaml/volleyball/demo.yaml
```

### 4.2 VideoCoAtt dataset
```
python demo_ours.py yaml/videocoatt/demo.yaml
```
