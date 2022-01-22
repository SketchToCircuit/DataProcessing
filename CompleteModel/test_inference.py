import os
from turtle import position
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from typing import List, Tuple
import cv2
import numpy as np
from object_detection.utils import label_map_util
import base64

import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import tensorboard

tf.config.optimizer.set_jit(True)

physical_devices = tf.config.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("No GPU detected!")

from combined_model import CombinedModel

from pin_model import PinDetectionModel

pins = tf.constant([[0.8, 0.3], [0.8, 0.7], [0.2, 0.5], [0.3, 0.7]])
pin_vals = tf.constant([0.7, 0.6, 0.4, 0.4])
class_id = tf.constant(27, tf.int32)
print(PinDetectionModel._assert_correct_pin_count(pins, pin_vals, class_id))
exit()

# from detection_model import ObjectDetectionModel
# det_model = ObjectDetectionModel('./ObjectDetection/exported_models/ssd_resnet101_640_v11/saved_model')

# writer = tf.summary.create_file_writer('./CompleteModel/Tensorboard')
# tf.summary.trace_on(graph=True, profiler=True)
# detections = det_model(tf.zeros((1, 640, 640, 3), tf.uint8))
# with writer.as_default():
#   tf.summary.trace_export(
#       name="DetectionModel",
#       step=0,
#       profiler_outdir='./CompleteModel/Tensorboard')
# exit()

with open('./CompleteModel/test.jpeg', "rb") as f:
    img_encoded = base64.b64encode(f.read())

# orig_img = cv2.imread('./CompleteModel/test.jpeg')
# img = tf.convert_to_tensor(orig_img, dtype=tf.uint8)

model = CombinedModel('./ObjectDetection/exported_models/ssd_resnet101_640_v11/saved_model', './PinDetection/exported/1')

classes, sample_indices, pins, pin_cmp_ids = model(img_encoded).values()
print(pin_cmp_ids)

# for i in range(tf.shape(patches)[0].numpy()):
#     patch = (patches[i].numpy()*255).astype(np.uint8)
#     patch = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)

#     positions = [peak_pos.numpy()[j] * patch.shape[0] for j in range(tf.shape(batch_ind)[0].numpy()) if batch_ind[j].numpy() == i]

#     for pos in positions:
#         patch = cv2.circle(patch, np.round(pos).astype(int), 3, (255, 0, 0), cv2.FILLED)
#     cv2.imwrite(f'./CompleteModel/test_patches_{i}.jpeg', patch)

signature = {tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY : model.__call__.get_concrete_function(tf.TensorSpec((None), dtype=tf.string))}
tf.saved_model.save(model, './CompleteModel/Exported', signature)

exit()

for det_class, box, idx in zip(classes.numpy(), boxes.numpy(), img_indices.numpy()):
    orig_img = cv2.rectangle(orig_img, (box[1], box[0]), (box[3], box[2]), (0, 0, 255), thickness=2)
    
cv2.imwrite('./CompleteModel/test_detected.jpeg', orig_img)

# inference_function = InferenceModel()
# concrete_fun = inference_function.__call__.get_concrete_function(tf.TensorSpec((640, 640, 3), tf.uint8))
# frozen = convert_variables_to_constants_v2(concrete_fun)
# graph_def = frozen.graph.as_graph_def()

# print(frozen.inputs)
# print(frozen.outputs)