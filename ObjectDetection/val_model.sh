#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate neural
export CUDA_VISIBLE_DEVICES=-1
python model_main_tf2.py \
--pipeline_config_path=./models/${MODEL_NAME}/${MODEL_VERSION}/pipeline.config \
--model_dir=./models/${MODEL_NAME}/${MODEL_VERSION}/ \
--checkpoint_dir=./models/${MODEL_NAME}/${MODEL_VERSION}/ \
--sample_1_of_n_eval_examples=1