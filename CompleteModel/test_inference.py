import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from typing import List, Tuple
import cv2
import numpy as np
from object_detection.utils import label_map_util

import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

tf.config.optimizer.set_jit(True)

physical_devices = tf.config.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("No GPU detected!")

from img_bbox_processing import MergeBoxes, SplitImage
from detection_model import ObjectDetectionModel

NUM_CLASSES = 42 # without background

category_index = label_map_util.create_category_index_from_labelmap('./ObjectDetection/data/label_map.pbtxt',use_display_name=True)

orig_img = cv2.imread('./CompleteModel/test.jpeg')

img = tf.convert_to_tensor(orig_img, dtype=tf.uint8)
images, offsets, sub_size = SplitImage(result_size=640, max_sub_size=800, overlap=100)(img)

sf = tf.reduce_max(sub_size / 640).numpy()

for i, patch in enumerate(images.numpy()):
    cv2.imwrite(f'./CompleteModel/test_patch_{i}.jpeg', patch)

boxes, classes, img_indices = ObjectDetectionModel('./ObjectDetection/exported_models/ssd_resnet101_640_v8/saved_model')(images).values()

boxes = MergeBoxes(src_size=640)(boxes)

for det_class, box, idx in zip(classes.numpy(), boxes.numpy(), img_indices.numpy()):
    box = box.numpy()
    orig_img = cv2.rectangle(orig_img, (box[1], box[0]), (box[3], box[2]), (0, 0, 255), thickness=2)
    
cv2.imwrite('./CompleteModel/test_detected.jpeg', orig_img)

# inference_function = InferenceModel()
# concrete_fun = inference_function.__call__.get_concrete_function(tf.TensorSpec((640, 640, 3), tf.uint8))
# frozen = convert_variables_to_constants_v2(concrete_fun)
# graph_def = frozen.graph.as_graph_def()

# print(frozen.inputs)
# print(frozen.outputs)