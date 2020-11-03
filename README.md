# hip_grade_deep_learning
code for the paper: clinical grade of hip x-ray radiographs using deep learning

## Setup

### RequireMents

```bash
pythont 3.7
```


## Running

### Setup

All commands must be ran from the project root. The following environment variables must be defined:
```bash
export ML_DATA="path to where you want the datasets saved"
export PYTHONPATH=$PYTHONPATH:.
```

### Example

For example, training a remixmatch with 32 filters and 4 augmentations on cifar10 shuffled with `seed=3`, 250 labeled samples and 5000
validation samples:
```bash
CUDA_VISIBLE_DEVICES=0 python cta/cta_remixmatch.py --filters=32 --K=4 --dataset=cifar10.3@250-5000 --w_match=1.5 --beta=0.75 --train_dir ./experiments/remixmatch
```

Available labelled sizes are 40, 100, 250, 1000, 4000.
For validation, available sizes are 1, 5000.
Possible shuffling seeds are 1, 2, 3, 4, 5 and 0 for no shuffling (0 is not used in practiced since data requires to be
shuffled for gradient descent to work properly).


## Citing this work

```
@article{berthelot2019remixmatch,
    title={ReMixMatch: Semi-Supervised Learning with Distribution Alignment and Augmentation Anchoring},
    author={David Berthelot and Nicholas Carlini and Ekin D. Cubuk and Alex Kurakin and Kihyuk Sohn and Han Zhang and Colin Raffel},
    journal={arXiv preprint arXiv:1911.09785},
    year={2019},
}
```
