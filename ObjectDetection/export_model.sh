#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate neural
export CUDA_VISIBLE_DEVICES=-1
export MODEL_VERSION=v14
python exporter_main_v2.py \
--pipeline_config_path=./models/ssd_resnet101_640/${MODEL_VERSION}/pipeline.config \
--trained_checkpoint_dir=./models/ssd_resnet101_640/${MODEL_VERSION}/best_ckps \
--output_directory=./exported_models/ssd_resnet101_640_${MODEL_VERSION}/ \
--input_type=float_image_tensor \
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
                box_predictor { \
                  weight_shared_convolutional_box_predictor { \
                    use_dropout: false \
                  } \
                } \
              } \
            }"