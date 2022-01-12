#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate neural
export CUDA_VISIBLE_DEVICES=-1
python exporter_main_v2.py \
--pipeline_config_path=./models/ssd_resnet101_640/v8/pipeline.config \
--trained_checkpoint_dir=./models/ssd_resnet101_640/v8/best_ckps \
--output_directory=./exported_models/ssd_resnet101_640_v8/ \
--input_type=float_image_tensor
--config_override " \
            model { \
              ssd { \
                image_resizer { \
                  fixed_shape_resizer { \
                    height: 640 \
                    width: 640 \
                    resize_method: AREA \
                  } \
                } \
              } \
            }"