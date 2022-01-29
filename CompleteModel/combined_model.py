import tensorflow as tf

from img_bbox_processing import MergeBoxes, SplitImage, resize_and_pad, savely_decode_base64
from detection_model import ObjectDetectionModel
from pin_model import PinDetectionModel

NUM_CLASSES = 42 # without background

class CombinedModel(tf.Module):
    def __init__(self, object_model_path, pin_model_path, patch_size=640, object_size=64):
        self._obj_model = ObjectDetectionModel(object_model_path)
        self._pin_model = PinDetectionModel(pin_model_path)
        self._patch_size: int = patch_size
        self._object_size = object_size

    @tf.function(input_signature=[tf.TensorSpec((None), dtype=tf.string)], experimental_follow_type_hints=True)
    def __call__(self, image: tf.Tensor):
        img = savely_decode_base64(image)
        
        with tf.name_scope('split_orig_image'):
            images, offsets, sub_size = SplitImage(result_size=self._patch_size, max_sub_size=int(self._patch_size*1.25), overlap=100)(img)
        
        with tf.name_scope('object_detection'):
            boxes, img_indices, class_probabilities = self._obj_model(images).values()

        with tf.name_scope('merge_boxes'):
            boxes = MergeBoxes(src_size=self._patch_size)(boxes, img_indices, offsets, sub_size)

        num_detections = tf.shape(boxes)[0]
        patches = tf.TensorArray(tf.float32, size=num_detections, element_shape=tf.TensorShape((self._object_size, self._object_size, 1)))
        unscaled_sizes = tf.TensorArray(tf.float32, size=num_detections, element_shape=tf.TensorShape(()))
        patch_offsets = tf.TensorArray(tf.float32, size=num_detections, element_shape=tf.TensorShape((2)))

        with tf.name_scope('extract_objects'):
            for i in tf.range(num_detections):
                patch = img[boxes[i][0]:boxes[i][2], boxes[i][1]:boxes[i][3], 0]

                orig_size = tf.cast(tf.shape(patch)[:2], tf.float32)
                unscaled_sizes = unscaled_sizes.write(i, tf.reduce_max(orig_size))
                patch_offsets = patch_offsets.write(i, (tf.constant([1.0, 1.0]) - orig_size / tf.reduce_max(orig_size)) / 2.0)

                patch = tf.cast(resize_and_pad(tf.expand_dims(patch, -1), self._object_size), tf.float32) / 255.0
                patches = patches.write(i, patch)

        with tf.name_scope('pin_detection'):
            classes, sample_indices, pins, pin_cmp_ids = self._pin_model(patches.stack(), class_probabilities, unscaled_sizes.stack(), patch_offsets.stack()).values()
        
        pins = tf.cast(pins, tf.int32) + tf.gather(boxes, tf.gather(sample_indices, pin_cmp_ids))[:, 1::-1]
        boxes = tf.gather(boxes, sample_indices)

        return {'classes': tf.identity(classes, 'classes'), 'boxes': tf.identity(boxes, 'boxes'), 'pins': tf.identity(pins, 'pins'), 'pin_cmp_ids': tf.identity(pin_cmp_ids, 'pin_cmp_ids')}