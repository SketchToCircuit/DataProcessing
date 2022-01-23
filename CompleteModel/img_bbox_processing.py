from typing import Tuple
import tensorflow as tf

def erode(img, size):
    # https://stackoverflow.com/questions/54686895/tensorflow-dilation-behave-differently-than-morphological-dilation

    # create kernel with random size with shape (a, a, 1)
    size = tf.repeat(size, 2)
    size = tf.pad(size, paddings=tf.constant([[0, 1]]), constant_values=1)
    kernel = tf.ensure_shape(tf.zeros(size, dtype=tf.float32), [None, None, 1])

    img = tf.nn.erosion2d(img, filters=kernel, strides=(1,1,1,1), dilations=(1,1,1,1), padding="SAME", data_format="NHWC")

    return img

def savely_decode_base64(base64: tf.Tensor):
    # convert to web-safe base64
    base64 = tf.strings.regex_replace(base64, '\+', '-')
    base64 = tf.strings.regex_replace(base64, '\/', '_')

    raw = tf.io.decode_base64(base64)

    img = tf.io.decode_image(raw, expand_animations=False)

    if tf.shape(img)[2] != 3:
        img = tf.repeat(tf.expand_dims(img[:, :, 0], axis=-1), 3, axis=2)
    
    return tf.ensure_shape(img, (None, None, 3), name='img')

def resize_and_pad(img, size):
    img = tf.cast(255 - tf.image.resize_with_pad(255 - img, size, size, method=tf.image.ResizeMethod.AREA), tf.uint8)
    return img

class MergeBoxes():
    def __init__(self, src_size=640) -> None:
        self._src_size = src_size

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, 4), dtype=tf.float32), tf.TensorSpec(shape=(None,), dtype=tf.int32), tf.TensorSpec(shape=(None, 2), dtype=tf.int32), tf.TensorSpec(shape=(2), dtype=tf.int32)])
    def __call__(self, boxes, img_indices, offsets, sub_size):
        sf = tf.cast(tf.reduce_max(sub_size / self._src_size), tf.float32)
        per_patch_pad_offset = tf.clip_by_value(640 - tf.cast(sub_size, tf.float32) / sf, 0.0, self._src_size) / 2.0
        per_patch_pad_offset = tf.tile(per_patch_pad_offset, [2])

        patch_positions = tf.ensure_shape(tf.gather(offsets, img_indices), (None, 2))
        patch_positions = tf.tile(patch_positions, [1, 2])

        boxes = boxes * tf.cast(self._src_size, tf.float32)
        boxes = boxes - per_patch_pad_offset
        boxes = boxes * sf
        boxes = boxes + tf.cast(patch_positions, tf.float32)

        return tf.cast(boxes, tf.int32, name='boxes')

class SplitImage():
    def __init__(self, result_size=640, max_sub_size=800, overlap=100) -> None:
        self.result_size = result_size
        self.max_sub_size = max_sub_size
        self.overlap = overlap

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8)])
    def __call__(self, img: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
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
            return (tf.expand_dims(resize_and_pad(img, self.result_size), 0, name='images'), tf.constant([[0, 0]], dtype=tf.int32, name='offsets'), tf.identity(size[:2], name='orig_sub_size'))

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

                padded = resize_and_pad(patch, self.result_size)
                images = images.write(i_y + i_x * num[0], padded)

        return (images.stack(name='images'), offsets.stack(name='offsets'), tf.identity(sub_size, name='orig_sub_size'))