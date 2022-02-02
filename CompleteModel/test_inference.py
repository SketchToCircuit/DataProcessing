import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# set path to cupti
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64'

# magic optimization
os.environ['CUDA_CACHE_DISABLE'] = '0'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

import cv2
import numpy as np
import base64
import time

from scipy import signal

import tensorflow as tf

physical_devices = tf.config.list_physical_devices("GPU")
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("No GPU detected!")

tf.config.optimizer.set_experimental_options({
    'layout_optimizer': True,
    'constant_folding': True,
    'shape_optimization': True,
    'remapping': True,
    'arithmetic_optimization': True,
    'dependency_optimization': True,
    'loop_optimization': True,
    'function_optimization': True,
    'scoped_allocator_optimization': True,
    'implementation_selector': True,
    'disable_meta_optimizer': False
})

img = cv2.imread('./CompleteModel/test.jpeg', cv2.IMREAD_COLOR)

def normalize_avg_line_thickness(img, goal_thickness=4):
    _, bw = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), 127, 255, cv2.THRESH_BINARY_INV)
    dist = cv2.distanceTransform(bw, cv2.DIST_L2, 3)
    maxima = signal.argrelextrema(dist, np.greater, order=2)
    med_thick = np.median(dist[maxima]) * 2.0
    sf = goal_thickness / med_thick
    resized = cv2.resize(img, None, fx=sf, fy=sf, interpolation=cv2.INTER_AREA)
    return cv2.threshold(cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY), 0, 255, cv2.THRESH_OTSU)[1]

img = normalize_avg_line_thickness(img, 4)

from combined_model import CombinedModel

model = CombinedModel('./ObjectDetection/exported_models/ssd_resnet101_640_v14/saved_model', './PinDetection/exported/1')

classes, boxes, pins, pin_cmp_ids = model(base64.urlsafe_b64encode(cv2.imencode('.jpg', img)[1])).values()

colored_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

for box in boxes.numpy():
    colored_img = cv2.rectangle(colored_img, (box[1], box[0]), (box[3], box[2]), (0, 0, 255), thickness=2)

for pin in pins.numpy():
    colored_img = cv2.circle(colored_img, pin, 3, (255, 0, 0), thickness=cv2.FILLED)

cv2.imwrite('./CompleteModel/test_detected.jpeg', colored_img)

# Profiling
# before = time.time()
# for i in range(20):
#     classes, boxes, pins, pin_cmp_ids = model(base64.urlsafe_b64encode(cv2.imencode('.jpg', image)[1])).values()
# duration = time.time() - before
# print(f'{duration / 20.0}s per image!')