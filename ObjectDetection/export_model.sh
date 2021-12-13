#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate neural
python exporter_main_v2.py \
--pipeline_config_path=./models/ssd_resnet101_640/v6/pipeline.config \
--trained_checkpoint_dir=./models/ssd_resnet101_640/v6/ \
--output_directory=./exported_models/ssd_resnet101_640_v6/ \
--input_type=image_tensor