#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate neural
export CUDA_VISIBLE_DEVICES=-1
python model_main_tf2.py \
--pipeline_config_path=./models/ssd_resnet101_640/v7/pipeline.config \
--model_dir=./models/ssd_resnet101_640/v7/ \
--checkpoint_dir=./models/ssd_resnet101_640/v7/ \
--sample_1_of_n_eval_examples=1