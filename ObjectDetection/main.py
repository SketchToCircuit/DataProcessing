import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import cv2
import numpy as np

import utils.dataset_utils as ds_utils

for example in ds_utils.read_parse_tfrecords('train.tfrecord'):
    cv2.imshow('', example['image/decoded'].numpy())
    cv2.waitKey(0)
    cv2.destroyAllWindows()