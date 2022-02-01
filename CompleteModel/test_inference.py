import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# enable XLA for the CPU
os.environ['TF_XLA_FLAGS'] = '--tf_xla_cpu_global_jit'

# set path to cupti
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64'

# magic optimization
os.environ['CUDA_CACHE_DISABLE'] = '0'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

import cv2
import numpy as np
import base64

import tensorflow as tf

physical_devices = tf.config.list_physical_devices("GPU")
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("No GPU detected!")

tf.config.optimizer.set_jit(True)
tf.config.optimizer.set_experimental_options({
    'layout_optimizer': True,
    'constant_folding': True,
    'shape_optimization': True,
    'remapping': True,
    'arithmetic_optimization': True,
    'dependency_optimization': True,
    'loop_optimization': True,
    'function_optimization': True,
    'debug_stripper': True,
    'scoped_allocator_optimization': True,
    'implementation_selector': True,
    'disable_meta_optimizer': False
})

from combined_model import CombinedModel

with open('./CompleteModel/test2.jpeg', "rb") as f:
    img_encoded = base64.b64encode(f.read())

model = CombinedModel('./ObjectDetection/exported_models/ssd_resnet101_640_v14/saved_model', './PinDetection/exported/1')

# writer = tf.summary.create_file_writer('./CompleteModel/Tensorboard')
# tf.summary.trace_on(graph=True, profiler=True)

classes, boxes, pins, pin_cmp_ids = model(tf.constant(img_encoded)).values()

# with writer.as_default():
#   tf.summary.trace_export(
#       name="CompleteModel",
#       step=0,
#       profiler_outdir='./CompleteModel/Tensorboard')

img = cv2.imread('./CompleteModel/test2.jpeg', cv2.IMREAD_COLOR)

for box in boxes.numpy():
    img = cv2.rectangle(img, (box[1], box[0]), (box[3], box[2]), (0, 0, 255), thickness=2)

for pin in pins.numpy():
    img = cv2.circle(img, pin, 3, (255, 0, 0), thickness=cv2.FILLED)

cv2.imwrite('./CompleteModel/test2_detected.jpeg', img)

# Profiling
print('Start profiling!')
tf.profiler.experimental.start('./CompleteModel/Tensorboard')
for i in range(10):
    classes, boxes, pins, pin_cmp_ids = model(tf.constant(img_encoded)).values()
tf.profiler.experimental.stop()