import tensorflow as tf

class EnsembleAugmentor(tf.Module):
    def __init__(self):
        super().__init__(name='EnsembleAugmentor')

    @tf.function(input_signature=[tf.TensorSpec((None, 640, 640, 3), dtype=tf.uint8)], experimental_follow_type_hints=True)
    def augment(self, images: tf.Tensor):
        num_orig = tf.shape(images)[0]
        images = tf.concat([images, tf.image.rot90(images, 1), tf.image.rot90(images, 2), tf.image.rot90(images, 3)], axis=0)
        return images, num_orig
    
    @tf.function(input_signature=[tf.TensorSpec(shape=(None, 4), dtype=tf.float32), tf.TensorSpec(shape=(None,), dtype=tf.int32), tf.TensorSpec(shape=(None), dtype=tf.int32)], experimental_follow_type_hints=True)
    def reverse(self, boxes: tf.Tensor, img_indices: tf.Tensor, num_orig_imges: tf.Tensor):

        unmod_idx = tf.reduce_max(tf.where(img_indices < num_orig_imges)) + 1
        num_1_idx = tf.reduce_max(tf.where(img_indices < num_orig_imges * 2)) + 1
        num_2_idx = tf.reduce_max(tf.where(img_indices < num_orig_imges * 3)) + 1

        orig_boxes = boxes[:unmod_idx]
        boxes_1 = boxes[unmod_idx:num_1_idx]    # rotated 90deg ccw
        boxes_2 = boxes[num_1_idx:num_2_idx]    # rotated 180deg ccw
        boxes_3 = boxes[num_2_idx:]             # rotated 270deg ccw

        boxes_1 = tf.stack([boxes_1[:, 1], 1.0 - boxes_1[:, 0], boxes_1[:, 3], 1.0 - boxes_1[:, 2]], axis=1)    # swap x and y and then flip x
        boxes_2 = tf.constant([1.0, 1.0, 1.0, 1.0]) - boxes_2                                                   # flip points in both directions
        boxes_3 = tf.stack([1.0 - boxes_3[:, 1], boxes_3[:, 0], 1.0 - boxes_3[:, 3], boxes_3[:, 2]], axis=1)    # swap x and y and then flip y

        boxes = tf.concat([orig_boxes, boxes_1, boxes_2, boxes_3], axis=0)

        # ensure that bbox points are top left and bottom right corner
        boxes = tf.stack([tf.minimum(boxes[:, 0], boxes[:, 2]), tf.minimum(boxes[:, 1], boxes[:, 3]), tf.maximum(boxes[:, 0], boxes[:, 2]), tf.maximum(boxes[:, 1], boxes[:, 3])], axis=1)

        img_indices = tf.math.mod(img_indices, num_orig_imges)

        return boxes, img_indices