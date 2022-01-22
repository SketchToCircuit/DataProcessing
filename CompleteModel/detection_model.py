import tensorflow as tf
import numpy as np
from typing import Tuple, List

class ObjectDetectionModel(tf.Module):
    def __init__(self, saved_model_path, num_classes = 42):
        super().__init__(name='ObjectDetectionModel')
        self._num_classes = num_classes
        self._model = tf.saved_model.load(saved_model_path)
        self._other_obj_mask, self._combined_object_masks = self._get_combined_object_masks()

    @tf.function(input_signature=[tf.TensorSpec((None, 640, 640, 3), dtype=tf.uint8)], experimental_follow_type_hints=True)
    def __call__(self, imges: tf.Tensor):
        # model is exported for float-images, because uint8 datatype doesn't support batched inference.
        detections = self._model(tf.cast(imges, tf.float32))

        # normalize multiclass scores (inverse sigmoid -> softmax)
        multiclass_scores = detections['detection_multiclass_scores']
        multiclass_scores = -tf.math.log(1.0 / multiclass_scores - 1.0)
        multiclass_scores = tf.nn.softmax(multiclass_scores)

        # recreate scores based on selected objects and scores and make sure, it sums to 1
        orig_redistributed_scores = tf.where(tf.one_hot(tf.cast(detections['detection_classes'], dtype=tf.int32), self._num_classes + 1, dtype=tf.int32) == 1,
            tf.expand_dims(detections['detection_scores'], axis=-1),
            tf.expand_dims((1.0 - detections['detection_scores']) / self._num_classes, axis=-1))

        # combine original scores and selected scores
        multiclass_scores = multiclass_scores * 0.3 + orig_redistributed_scores * 0.7

        # choose single object scores
        combined_scores = tf.squeeze(tf.gather(multiclass_scores, tf.where(self._other_obj_mask == 1), axis=2), axis=-1)

        # combine scores of similar objects
        for mask in self._combined_object_masks:
            masked_scores = tf.gather(multiclass_scores, tf.where(mask == 1), axis=2)
            score = tf.reduce_sum(masked_scores, axis=2)
            combined_scores = tf.concat([combined_scores, score], axis=2)

        indices = tf.squeeze(tf.where(tf.reduce_max(combined_scores, axis=2) > 0.5))

        boxes = tf.ensure_shape(tf.gather_nd(detections['detection_boxes'], indices), (None, 4))
        #classes = tf.cast(tf.gather_nd(detections['detection_classes'], indices), tf.uint8)
        class_probabilities = tf.gather_nd(detections['detection_multiclass_scores'], indices)
        class_probabilities = -tf.math.log(1.0 / class_probabilities - 1.0)
        class_probabilities = tf.nn.softmax(class_probabilities)

        return {'boxes': boxes, 'img_indices': tf.cast(indices[:, 0], tf.int32), 'class_probabilities': class_probabilities[:, 1:]}

    def _get_combined_object_masks(self) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        combined_object_masks = []

        circular_obj_mask = np.zeros(self._num_classes + 1)
        circular_obj_mask[np.array([1, 14, 15, 20, 21, 40, 37, 38, 41])] = 1
        combined_object_masks.append(tf.convert_to_tensor(circular_obj_mask, dtype=tf.int32))

        diode_obj_mask = np.zeros(self._num_classes + 1)
        diode_obj_mask[np.array([7, 8, 9, 19])] = 1
        combined_object_masks.append(tf.convert_to_tensor(diode_obj_mask, dtype=tf.int32))

        btn_obj_mask = np.zeros(self._num_classes + 1)
        btn_obj_mask[np.array([3, 4, 33, 34, 35, 29])] = 1
        combined_object_masks.append(tf.convert_to_tensor(btn_obj_mask, dtype=tf.int32))

        mfet_obj_mask = np.zeros(self._num_classes + 1)
        mfet_obj_mask[np.array([22, 23, 24, 25])] = 1
        combined_object_masks.append(tf.convert_to_tensor(mfet_obj_mask, dtype=tf.int32))

        jfet_obj_mask = np.zeros(self._num_classes + 1)
        jfet_obj_mask[np.array([16, 17])] = 1
        combined_object_masks.append(tf.convert_to_tensor(jfet_obj_mask, dtype=tf.int32))

        bi_obj_mask = np.zeros(self._num_classes + 1)
        bi_obj_mask[np.array([27, 30])] = 1
        combined_object_masks.append(tf.convert_to_tensor(bi_obj_mask, dtype=tf.int32))

        r_obj_mask = np.zeros(self._num_classes + 1)
        r_obj_mask[np.array([31, 32, 18, 10])] = 1
        combined_object_masks.append(tf.convert_to_tensor(r_obj_mask, dtype=tf.int32))

        bat_obj_mask = np.zeros(self._num_classes + 1)
        bat_obj_mask[np.array([2, 39])] = 1
        combined_object_masks.append(tf.convert_to_tensor(bat_obj_mask, dtype=tf.int32))

        gnd_obj_mask = np.zeros(self._num_classes + 1)
        gnd_obj_mask[np.array([11, 13])] = 1
        combined_object_masks.append(tf.convert_to_tensor(gnd_obj_mask, dtype=tf.int32))

        c_obj_mask = np.zeros(self._num_classes + 1)
        c_obj_mask[np.array([5, 6])] = 1
        combined_object_masks.append(tf.convert_to_tensor(c_obj_mask, dtype=tf.int32))

        other_obj_mask = tf.zeros((self._num_classes + 1), dtype=tf.int32)
        for mask in combined_object_masks:
            other_obj_mask = other_obj_mask + mask
        other_obj_mask = 1 - other_obj_mask

        return other_obj_mask, combined_object_masks