import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import cv2
import numpy as np
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

with open('./CompleteModel/test.jpeg', "rb") as f:
    img_encoded = base64.b64encode(f.read())

model = CombinedModel('./ObjectDetection/exported_models/ssd_resnet101_640_v11/saved_model', './PinDetection/exported/1')

signature = {tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY : model.__call__.get_concrete_function(tf.TensorSpec((None), dtype=tf.string))}
tf.saved_model.save(model, './CompleteModel/Exported', signature)

writer = tf.summary.create_file_writer('./CompleteModel/Tensorboard')
tf.summary.trace_on(graph=True, profiler=True)

img, classes, boxes, pins, pin_cmp_ids = model(img_encoded).values()

with writer.as_default():
  tf.summary.trace_export(
      name="CompleteModel",
      step=0,
      profiler_outdir='./CompleteModel/Tensorboard')

img = img.numpy()

for box in boxes.numpy():
    img = cv2.rectangle(img, (box[1], box[0]), (box[3], box[2]), (0, 0, 255), thickness=2)

for pin in pins.numpy():
    img = cv2.circle(img, pin, 3, (255, 0, 0), thickness=cv2.FILLED)

cv2.imwrite('./CompleteModel/test_detected.jpeg', img)