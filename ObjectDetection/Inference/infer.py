import os

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf

physical_devices = tf.config.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("No GPU detected!")

import cv2
import numpy as np

from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import label_map_util

category_index = label_map_util.create_category_index_from_labelmap('./ObjectDetection/data/label_map.pbtxt',use_display_name=True)

orig_img = cv2.imread('./ObjectDetection/Inference/Images/test.jpeg')

sf = 640 / np.max(orig_img.shape)
orig_img = cv2.resize(orig_img, dsize=None, fx=sf, fy=sf, interpolation=cv2.INTER_AREA)

padded_img = np.full((640, 640, 3), 255, dtype=np.uint8)
padded_img[:orig_img.shape[0], :orig_img.shape[1]] = orig_img

# filter blue grid
img = np.where(np.all(padded_img < 128, axis=-1), 0, 255)

img = tf.expand_dims(tf.convert_to_tensor(img, dtype=tf.uint8), axis=-1)
img = tf.repeat(img, 3, axis=-1)
img = tf.expand_dims(img, axis=0)

model = tf.saved_model.load('./ObjectDetection/exported_models/ssd_resnet101_640_v6/saved_model')
detections = model(img)

for k in detections.keys():
    detections[k] = tf.squeeze(detections[k], 0)

indices = np.squeeze(np.argwhere(tf.squeeze(detections['detection_scores'],).numpy() > 0.5))

detected_img = orig_img.copy()
for box, label in zip(detections['detection_boxes'].numpy()[indices], detections['detection_classes'].numpy()[indices]):
    box = (box * 640).astype(int)
    detected_img = cv2.rectangle(detected_img, (box[1], box[0]), (box[3], box[2]), (0, 0, 255), thickness=2)
    
cv2.imwrite('./ObjectDetection/Inference/Images/test_detected.jpeg', detected_img)