import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

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

NUM_CLASSES = 42 # without background

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

combined_object_masks = []

circular_obj_mask = np.zeros(NUM_CLASSES + 1)
circular_obj_mask[np.array([1, 14, 15, 20, 21, 40, 37, 38, 41])] = 1
combined_object_masks.append(tf.convert_to_tensor(circular_obj_mask, dtype=tf.int32))

diode_obj_mask = np.zeros(NUM_CLASSES + 1)
diode_obj_mask[np.array([7, 8, 9, 19])] = 1
combined_object_masks.append(tf.convert_to_tensor(diode_obj_mask, dtype=tf.int32))

btn_obj_mask = np.zeros(NUM_CLASSES + 1)
btn_obj_mask[np.array([3, 4, 33, 34, 35, 29])] = 1
combined_object_masks.append(tf.convert_to_tensor(btn_obj_mask, dtype=tf.int32))

mfet_obj_mask = np.zeros(NUM_CLASSES + 1)
mfet_obj_mask[np.array([22, 23, 24, 25])] = 1
combined_object_masks.append(tf.convert_to_tensor(mfet_obj_mask, dtype=tf.int32))

jfet_obj_mask = np.zeros(NUM_CLASSES + 1)
jfet_obj_mask[np.array([16, 17])] = 1
combined_object_masks.append(tf.convert_to_tensor(jfet_obj_mask, dtype=tf.int32))

bi_obj_mask = np.zeros(NUM_CLASSES + 1)
bi_obj_mask[np.array([27, 30])] = 1
combined_object_masks.append(tf.convert_to_tensor(bi_obj_mask, dtype=tf.int32))

r_obj_mask = np.zeros(NUM_CLASSES + 1)
r_obj_mask[np.array([31, 32, 18, 10])] = 1
combined_object_masks.append(tf.convert_to_tensor(r_obj_mask, dtype=tf.int32))

bat_obj_mask = np.zeros(NUM_CLASSES + 1)
bat_obj_mask[np.array([2, 39])] = 1
combined_object_masks.append(tf.convert_to_tensor(bat_obj_mask, dtype=tf.int32))

gnd_obj_mask = np.zeros(NUM_CLASSES + 1)
gnd_obj_mask[np.array([11, 13])] = 1
combined_object_masks.append(tf.convert_to_tensor(gnd_obj_mask, dtype=tf.int32))

c_obj_mask = np.zeros(NUM_CLASSES + 1)
c_obj_mask[np.array([5, 6])] = 1
combined_object_masks.append(tf.convert_to_tensor(c_obj_mask, dtype=tf.int32))

other_obj_mask = tf.zeros((NUM_CLASSES + 1), dtype=tf.int32)
for mask in combined_object_masks:
    other_obj_mask = other_obj_mask + mask
other_obj_mask = 1 - other_obj_mask

# load model and make prediction
model = tf.saved_model.load('./ObjectDetection/exported_models/ssd_resnet101_640_v8/saved_model')
detections = model(img)

# squeeze batch dimension
for k in detections.keys():
    detections[k] = tf.squeeze(detections[k], 0)

scores = detections['detection_multiclass_scores']
# combine scores of similar objects
combined_scores = tf.squeeze(tf.gather(scores, tf.where(other_obj_mask == 1), axis=1), axis=-1)

for mask in combined_object_masks:
    combined_scores = tf.concat([combined_scores, tf.reduce_sum(tf.gather(scores, tf.where(mask == 1), axis=1), axis=1)], axis=1)

indices = np.squeeze(np.argwhere(tf.reduce_max(combined_scores, axis=1) > 0.5))

print(tf.gather(detections['detection_scores'], indices))
print(tf.gather(detections['detection_classes'], indices))

detected_img = orig_img.copy()
for box, in zip(detections['detection_boxes'].numpy()[indices]):
    box = (box * 640).astype(int)
    detected_img = cv2.rectangle(detected_img, (box[1], box[0]), (box[3], box[2]), (0, 0, 255), thickness=2)
    
cv2.imwrite('./ObjectDetection/Inference/Images/test_detected.jpeg', detected_img)