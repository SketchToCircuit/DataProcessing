import tensorflow as tf
import numpy as np
from typing import Tuple, List

class ObjectDetectionModel(tf.Module):
    def __init__(self, saved_model_path, num_classes = 42):
        super().__init__(name='ObjectDetectionModel')
        self._num_classes = num_classes
        self._model = tf.saved_model.load(saved_model_path)

    @tf.function(input_signature=[tf.TensorSpec((None, 640, 640, 3), dtype=tf.uint8)], experimental_follow_type_hints=True)
    def __call__(self, imges: tf.Tensor):
        # model is exported for float-images, because uint8 datatype doesn't support batched inference.
        detections = self._model(tf.cast(imges, tf.float32))

        indices = tf.squeeze(tf.where(detections['detection_scores'] > 0.2))

        boxes = tf.ensure_shape(tf.gather_nd(detections['detection_boxes'], indices), (None, 4))

        class_probabilities = tf.gather_nd(detections['detection_multiclass_scores'], indices)
        class_probabilities = -tf.math.log(1.0 / class_probabilities - 1.0)
        class_probabilities = tf.nn.softmax(class_probabilities)

        return boxes, tf.cast(indices[:, 0], tf.int32), class_probabilities[:, 1:]