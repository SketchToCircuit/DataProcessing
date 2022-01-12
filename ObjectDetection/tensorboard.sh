#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate neural
tensorboard --logdir ./models/${MODEL_NAME}/${MODEL_VERSION}