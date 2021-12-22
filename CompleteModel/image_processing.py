from typing import Tuple
import tensorflow as tf

def resize_and_pad_bottom_left(img, size):
    resized = tf.cast(tf.image.resize(img, size, preserve_aspect_ratio=True, method=tf.image.ResizeMethod.AREA, antialias=True), tf.uint8)
    res_size = tf.shape(resized)
    padded = tf.pad(resized, [[0, size[0] - res_size[0]], [0, size[1] - res_size[1]], [0, 0]], mode='CONSTANT', constant_values=255)
    return padded

class SplitImage():
    def __init__(self, result_size=640, max_sub_size=800, overlap=100) -> None:
        self.result_size = result_size
        self.max_sub_size = max_sub_size
        self.overlap = overlap

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8), ])
    def __call__(self, img: tf.Tensor, ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        '''
        input: image as tf.Tensor in RGB with shape (None, None, 3) and dtype=tf.uint8
        return: Tuple of images and original sizes plus offsets -> (images, orig_sizes, offsets)
            - images: tf.Tensor; shape=(None, result_size, result_size, 3); dtype=tf.uint8
            - offsets: tf.Tensor; shape=(None, 2); dtype=tf.int32 with y and x of upper left corners of patches
            - orig_sub_size: tf.Tensor; shape=(2); dtype=tf.tf.int32 with h and w of patches before scaling
            - all 'None' shapes are the same size
        '''
        size = tf.shape(img)

        if tf.reduce_all(size < self.max_sub_size):
            return (tf.expand_dims(resize_and_pad_bottom_left(img, [self.result_size, self.result_size]), 0, name='images'), tf.constant([[0, 0]], dtype=tf.int32, name='offsets'), tf.identity(size[:2], name='orig_sub_size'))

        # for visualization of these numbers: https://drive.google.com/file/d/1aZhQrWoTZLHLKWHMVWnJfyu1em_SFwBW/view?usp=sharing
        num = tf.cast(tf.math.maximum(tf.math.ceil((size[:2] - self.overlap) / (self.max_sub_size - self.overlap)), 1), dtype=tf.int32)
        sub_size = tf.cast(tf.math.floor((size[:2] + (num - 1) * self.overlap) / num), dtype=tf.int32)
        stride = sub_size - self.overlap

        offsets = tf.TensorArray(tf.int32, size=num[0]*num[1], element_shape=tf.TensorShape((2)))
        images = tf.TensorArray(tf.uint8, size=num[0]*num[1], element_shape=tf.TensorShape((self.result_size, self.result_size, 3)))

        for i_y in tf.range(num[0]):
            for i_x in tf.range(num[1]):
                offset = stride * [i_y, i_x]
                offsets = offsets.write(i_y + i_x * num[0], offset)
                patch = img[offset[0]:offset[0]+sub_size[0], offset[1]:offset[1]+sub_size[1], :]

                padded = resize_and_pad_bottom_left(patch, [self.result_size, self.result_size])
                images = images.write(i_y + i_x * num[0], padded)

        return (images.stack(name='images'), offsets.stack(name='offsets'), tf.identity(sub_size, name='orig_sub_size'))