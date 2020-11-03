# hip_grade_deep_learning
code for clinical grade of hip x-ray radiographs using deep learning
Code for the paper: "[ReMixMatch: Semi-Supervised Learning with Distribution Alignment and Augmentation Anchoring](https://arxiv.org/abs/1911.09785)"


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


#### Multi-GPU training
Just pass more GPUs and remixmatch automatically scales to them, here we assign GPUs 4-7 to the program:
```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 python cta/cta_remixmatch.py --filters=32 --K=4 --dataset=cifar10.3@250-5000 --w_match=1.5 --beta=0.75 --train_dir ./experiments/remixmatch
```

### Valid dataset names
```bash
for dataset in cifar10 svhn svhn_noextra; do
for seed in 0 1 2 3 4 5; do
for valid in 1 5000; do
for size in 40 250 1000 4000; do
    echo "${dataset}.${seed}@${size}-${valid}"
done; done; done; done

for seed in 0 1 2 3 4 5; do
for valid in 1 5000; do
    echo "cifar100.${seed}@10000-${valid}"
done; done

for seed in 1 2 3 4 5; do
for valid in 1 5000; do
    echo "stl10.${seed}@1000-${valid}"
done; done
echo "stl10.1@5000-1"
```


## Monitoring training progress

You can point tensorboard to the training folder (by default it is `--train_dir=./experiments`) to monitor the training
process:

```bash
tensorboard.sh --port 6007 --logdir experiments
```

## Checkpoint accuracy

We compute the median accuracy of the last 20 checkpoints in the paper, this is done through this code:

```bash
# Following the previous example in which we trained cifar10.3@250-5000, extracting accuracy:
./scripts/extract_accuracy.py experiments/cifar10.d.d.d.3\@250-5000/CTAugment_depth2_th0.80_decay0.990/CTAReMixMatch_K4_archresnet_batch64_beta0.75_filters32_lr0.002_nclass10_redux1st_repeat4_scales3_use_dmTrue_use_xeTrue_w_kl0.5_w_match1.5_w_rot0.5_warmup_kimg1024_wd0.02/
# The command above will create a stats/accuracy.json file in the model folder.
# The format is JSON so you can either see its content as a text file or process it to your liking.
```

## Reproducing tables from the paper

Check the contents of the `runs/*.sh` files, these will give you the commands (and the hyper-parameters) to reproduce the results from the paper.

## Citing this work

```
@article{berthelot2019remixmatch,
    title={ReMixMatch: Semi-Supervised Learning with Distribution Alignment and Augmentation Anchoring},
    author={David Berthelot and Nicholas Carlini and Ekin D. Cubuk and Alex Kurakin and Kihyuk Sohn and Han Zhang and Colin Raffel},
    journal={arXiv preprint arXiv:1911.09785},
    year={2019},
}
```
