# hip_grade_deep_learning
code for the paper: clinical grade of hip x-ray radiographs using deep learning

## Setup

### Requirements

```bash
python 3.7.8
pytorch 1.4.0
CUDA 10.1
```


## Running

### Train model

train hip 7 classes model:
```bash
python main.py --model_id xception_hip --num_class 7 --bnm_loss 0 --bnm_loss_weight 0.0 --gpu 0,1 --root_path ../data/hip_7cls/training_data --train_file ../data/hip_7cls/training.txt --test_file ../data/hip_7cls/testing.txt --task hip_7cls
```
train onfh 3 classes model:
```bash
python main.py --model_id xception_onfh_fold1 --num_class 3  --bnm_loss 0 --bnm_loss_weight 0.0 --gpu 0,1
```
### Test model
download 7 classification model from https://drive.google.com/file/d/1nnEUXHUrxWpvVREKYv68_H2x8bS3Yzyf/view?usp=sharing

download 3classification model from https://drive.google.com/file/d/1Ens9erEWUtoXDuHM_OJ-GU0-nwiklHlI/view?usp=sharing

test hip 7cls model:
```bash
python test_7cls.py
```
test onfh 3cls model:
```bash
python test_3cls.py
```


## Citing this work

```
```
