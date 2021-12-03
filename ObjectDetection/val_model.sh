export CUDA_VISIBLE_DEVICES=-1
python ObjectDetection/model_main_tf2.py \
--pipeline_config_path=./ObjectDetection/models/ssd_resnet101_640/v1/pipeline.config \
--model_dir=./ObjectDetection/models/ssd_resnet101_640/v1/ \
--checkpoint_dir=./ObjectDetection/models/ssd_resnet101_640/v1/ \
--sample_1_of_n_eval_examples=1