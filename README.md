# Intro

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

* Volleyball dataset (dataset)  
https://github.com/mostafa-saad/deep-activity-rec

* Volleyball dataset (attributes)  
https://drive.google.com/drive/folders/1O55_wri92uv87g-2aDh8ll6dFVupmFaB?usp=share_link

* VideoCoAtt dataset (dataset)  
http://www.stat.ucla.edu/~lifengfan/shared_attention

* VideoCoAtt dataset (attributes)  
https://drive.google.com/drive/folders/1O55_wri92uv87g-2aDh8ll6dFVupmFaB?usp=share_link

## 2. Training
### 2.1 Volleyball dataset
```
python train.py yaml/volleyball/train_ours_p_p.yaml
python train.py yaml/volleyball/train_ours.yaml
```

### 2.2 VideoCoAtt dataset
```
python train.py yaml/videocoatt/train_ours_p_p.yaml
python train.py yaml/videocoatt/train_ours.yaml
```

## 3. Evaluation
### 3.1 Volleyball dataset
```
python eval_on_volleyball_ours.py yaml/volleyball/eval.yaml
```


### 3.2 VideoCoAtt dataset
```
python eval_on_videocoatt_ours.py yaml/videocoatt/eval.yaml
```

