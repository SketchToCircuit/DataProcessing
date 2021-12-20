import os
from typing import List, Tuple

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

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

class InferenceModel():
    def __init__(self):
        # load model
        self._model = tf.saved_model.load('./ObjectDetection/exported_models/ssd_resnet101_640_v8/saved_model')
        self._other_obj_mask, self._combined_object_masks = self._get_combined_object_masks()
    
    @tf.function
    def __call__(self, img: tf.Tensor):
        img = tf.expand_dims(img, axis=0)
    
        detections = self._model(img)
    
        multiclass_scores = tf.squeeze(detections['detection_multiclass_scores'], 0)
    
        # recreate scores based on selected objects and scores and make sure, it sums to 1
        orig_redistributed_scores = tf.where(tf.one_hot(tf.cast(tf.squeeze(detections['detection_classes'], 0), dtype=tf.int32), NUM_CLASSES + 1, dtype=tf.int32) == 1,
            tf.expand_dims(tf.squeeze(detections['detection_scores'], 0), axis=-1),
            tf.expand_dims((1.0 - tf.squeeze(detections['detection_scores'], 0)) / NUM_CLASSES, axis=-1))
    
        # combine original scores and selected scores
        multiclass_scores = multiclass_scores * 0.7 + orig_redistributed_scores * 0.3
    
        # choose single object scores
        combined_scores = tf.squeeze(tf.gather(multiclass_scores, tf.where(self._other_obj_mask == 1), axis=1), axis=-1)

        # combine scores of similar objects
        for mask in self._combined_object_masks:
            masked_scores = tf.gather(multiclass_scores, tf.where(mask == 1), axis=1)
            score = tf.reduce_sum(masked_scores, axis=1)
            combined_scores = tf.concat([combined_scores,score], axis=1)
    
        # normalize scores
        combined_scores = combined_scores / tf.reduce_sum(combined_scores, axis=1, keepdims=True)
    
        indices = tf.squeeze(tf.where(tf.reduce_max(combined_scores, axis=1) > 0.5))
        #indices = np.squeeze(np.argwhere(detections['detection_scores'] > 0.5))

        boxes = tf.ensure_shape(tf.gather(tf.squeeze(detections['detection_boxes'], 0), indices), (None, 4))
        classes = tf.cast(tf.gather(tf.squeeze(detections['detection_classes'], 0), indices), tf.uint8)

        return {'boxes': boxes, 'classes': classes}

    @staticmethod
    def _get_combined_object_masks() -> Tuple[tf.Tensor, List[tf.Tensor]]:
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

        return other_obj_mask, combined_object_masks

inference_function = InferenceModel()
concrete_fun = inference_function.__call__.get_concrete_function(tf.TensorSpec((640, 640, 3), tf.uint8))
frozen = convert_variables_to_constants_v2(concrete_fun)
graph_def = frozen.graph.as_graph_def()

print(frozen.inputs)
print(frozen.outputs)

boxes, classes = inference_function(img).values()
for det_class, box in zip(classes.numpy(), boxes.numpy()):
    box = (box * 640).astype(int)
    orig_img = cv2.rectangle(orig_img, (box[1], box[0]), (box[3], box[2]), (0, 0, 255), thickness=2)
    
cv2.imwrite('./ObjectDetection/Inference/Images/test_detected.jpeg', orig_img)