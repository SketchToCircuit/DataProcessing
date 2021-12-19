#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate neural
tensorboard --logdir ./models/ssd_resnet101_640/v8