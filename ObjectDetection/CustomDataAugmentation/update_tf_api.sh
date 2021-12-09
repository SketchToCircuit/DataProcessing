#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate neural
cp custom_augmentation.py ../../../TensorflowModels/models/research/object_detection/utils
cd ../../../TensorflowModels/models/research/
pip install --no-deps --upgrade .