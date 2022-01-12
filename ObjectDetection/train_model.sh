#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate neural
python model_main_tf2.py \
--pipeline_config_path=./models/${MODEL_NAME}/${MODEL_VERSION}/pipeline.config \
--model_dir=./models/${MODEL_NAME}/${MODEL_VERSION}/ \
--checkpoint_every_n=100 \
--alsologtostderr