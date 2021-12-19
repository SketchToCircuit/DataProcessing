#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate neural
python model_main_tf2.py \
--pipeline_config_path=./models/ssd_resnet101_640/v8/pipeline.config \
--model_dir=./models/ssd_resnet101_640/v8/ \
--checkpoint_every_n=100 \
--alsologtostderr