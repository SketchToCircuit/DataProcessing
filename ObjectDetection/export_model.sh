#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate neural
export CUDA_VISIBLE_DEVICES=-1
export MODEL_VERSION=v2
export MODEL_NAME=centernet_hourglass104_512
python exporter_main_v2.py \
--pipeline_config_path=./models/${MODEL_NAME}/${MODEL_VERSION}/pipeline.config \
--trained_checkpoint_dir=./models/${MODEL_NAME}/${MODEL_VERSION}/best_ckps \
--output_directory=./exported_models/${MODEL_NAME}_${MODEL_VERSION}/ \
--input_type=float_image_tensor \
--config_override " \
            model { \
              center_net { \
                image_resizer { \
                  fixed_shape_resizer { \
                    height: 512 \
                    width: 512 \
                    resize_method: AREA \
                  } \
                } \
              } \
            }"

# --config_override " \
#             model { \
#               ssd { \
#                 image_resizer { \
#                   fixed_shape_resizer { \
#                     height: 640 \
#                     width: 640 \
#                     resize_method: AREA \
#                   } \
#                 } \
#               } \
#             }"