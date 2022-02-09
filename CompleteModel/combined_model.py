from typing import Dict

import tensorflow as tf

from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

from img_bbox_processing import MergeBoxes, SplitImage, resize_and_pad
from detection_model import ObjectDetectionModel
from pin_model import PinDetectionModel
from ensemble_augmentation import EnsembleAugmentor
from box_fusion import FuseBoxes

NUM_CLASSES = 42 # without background

class CombinedModel(tf.Module):
    def __init__(self, object_model_path, pin_model_path, patch_size=640, object_size=64, hyperparameters=None, do_not_convert_variables=False):
        obj_model = ObjectDetectionModel(object_model_path)
        pin_model = PinDetectionModel(pin_model_path, hyperparameters=hyperparameters)

        if do_not_convert_variables:
            self._obj_model_func = obj_model
            self._pin_model_func = pin_model
        else:
            self._obj_model_func = convert_variables_to_constants_v2(obj_model.__call__.get_concrete_function())
            self._pin_model_func = convert_variables_to_constants_v2(pin_model.__call__.get_concrete_function())
        self._ensemble_augmentor = EnsembleAugmentor()
        self._box_fusion = FuseBoxes(hyperparameters)
        self._patch_size: int = patch_size
        self._object_size = object_size

    @tf.function(input_signature=[tf.TensorSpec((None), dtype=tf.string, name='image')], experimental_follow_type_hints=True)
    def __call__(self, image: tf.Tensor):
        raw = tf.io.decode_base64(image)
        image = tf.io.decode_image(raw, expand_animations=False)
        image = tf.image.grayscale_to_rgb(image)

        with tf.name_scope('split_orig_image'):
            images, offsets, sub_size = SplitImage(result_size=self._patch_size, max_sub_size=int(self._patch_size*1.25), overlap=100)(image)

        with tf.name_scope('ensemble_augmentation'):
            images, num_orig_imges = self._ensemble_augmentor.augment(images)

        with tf.name_scope('object_detection'):
            boxes, img_indices, class_probabilities = self._obj_model_func(images)

        with tf.name_scope('reverse_augmentation'):
            boxes, img_indices = self._ensemble_augmentor.reverse(boxes, img_indices, num_orig_imges)

        with tf.name_scope('merge_boxes'):
            boxes = MergeBoxes(src_size=self._patch_size)(boxes, img_indices, offsets, sub_size)

            # clip boxes to image size
            boxes = tf.maximum(boxes, 0)
            boxes = tf.minimum(boxes, tf.tile(tf.shape(image)[:2], [2]) - 1)

        with tf.name_scope('box_fusion'):
            boxes, class_probabilities = self._box_fusion(boxes, class_probabilities)

        num_detections = tf.shape(boxes)[0]
        patches = tf.TensorArray(tf.float32, size=0, dynamic_size=True, element_shape=tf.TensorShape((self._object_size, self._object_size, 1)))
        unscaled_sizes = tf.TensorArray(tf.float32, size=0, dynamic_size=True, element_shape=tf.TensorShape(()))
        patch_offsets = tf.TensorArray(tf.float32, size=0, dynamic_size=True, element_shape=tf.TensorShape((2)))

        with tf.name_scope('extract_objects'):
            for i in tf.range(num_detections):
                patch = image[boxes[i][0]:boxes[i][2]+1, boxes[i][1]:boxes[i][3]+1, 0]

                if tf.reduce_any(tf.shape(patch) <= 0):
                    tf.print('Invalid box!')
                    tf.print(boxes[i])
                else:
                    write_indx = patches.size()

                    orig_size = tf.cast(tf.shape(patch)[:2], tf.float32)
                    unscaled_sizes = unscaled_sizes.write(write_indx, tf.reduce_max(orig_size))
                    patch_offsets = patch_offsets.write(write_indx, (tf.constant([1.0, 1.0]) - orig_size / tf.reduce_max(orig_size)) / 2.0)

                    patch = tf.cast(resize_and_pad(patch[..., None], self._object_size), tf.float32) / 255.0
                    patches = patches.write(write_indx, patch)

        with tf.name_scope('pin_detection'):
            classes, sample_indices, pins, pin_cmp_ids = self._pin_model_func(patches.stack(), class_probabilities, unscaled_sizes.stack(), patch_offsets.stack())
            
            corresponding_boxes = tf.gather(boxes, tf.gather(sample_indices, pin_cmp_ids))
            pins = tf.cast(pins, tf.int32) + corresponding_boxes[:, 1::-1]

            # clip pin coordinates to bbox
            pins = tf.minimum(tf.maximum(pins, corresponding_boxes[:, 1::-1]), corresponding_boxes[:, 3:1:-1])
            boxes = tf.gather(boxes, sample_indices)
            
        return {'classes': tf.identity(classes, 'classes'), 'boxes': tf.identity(boxes, 'boxes'), 'pins': tf.identity(pins, 'pins'), 'pin_cmp_ids': tf.identity(pin_cmp_ids, 'pin_cmp_ids')}