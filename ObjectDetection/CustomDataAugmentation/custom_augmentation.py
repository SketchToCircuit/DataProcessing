import os

from tensorflow._api.v2 import data

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import cv2
import numpy as np
import math

@tf.function
def threshold(img):
    return tf.where(img < 200, 0.0, 255.0)

@tf.function
def dilate(img, size):
    # https://stackoverflow.com/questions/54686895/tensorflow-dilation-behave-differently-than-morphological-dilation
    # image needs another dimension for a batchszie of 1
    img = tf.expand_dims(img, axis=0)

    # max kernel_size = 10
    # create kernel with random size with shape (a, a, 3)
    size = tf.clip_by_value(size, 1, 10)
    size = tf.repeat(size, 2)
    size = tf.pad(size, paddings=tf.constant([[0, 1]]), constant_values=3)
    kernel = tf.zeros(size, dtype=tf.float32)

    img = tf.nn.dilation2d(img, filters=kernel, strides=(1,1,1,1), dilations=(1,1,1,1), padding="SAME", data_format="NHWC")

    # revert batch dimension
    img = tf.squeeze(img, axis=0)

    return img

@tf.function
def erode(img, size):
    # https://stackoverflow.com/questions/54686895/tensorflow-dilation-behave-differently-than-morphological-dilation
    # image needs another dimension for a batchszie of 1
    img = tf.expand_dims(img, axis=0)

    # max kernel_size = 10
    # create kernel with random size with shape (a, a, 3)
    size = tf.clip_by_value(size, 1, 10)
    size = tf.repeat(size, 2)
    size = tf.pad(size, paddings=tf.constant([[0, 1]]), constant_values=3)
    kernel = tf.zeros(size, dtype=tf.float32)

    img = tf.nn.erosion2d(img, filters=kernel, strides=(1,1,1,1), dilations=(1,1,1,1), padding="SAME", data_format="NHWC")

    # revert batch dimension
    img = tf.squeeze(img, axis=0)

    return img

@tf.function
def augment(image, boxes):
    '''
    image: Tensor("", shape=(None, None, 3), dtype=float32) with values in [0, 255]
    boxes: Tensor("", shape=(None, 4), dtype=float32) every item is in form of [ymin, xmin, ymax, xmax]
    '''
    image = threshold(image) # always binarize image

    # 50% dilation or erotion
    image = tf.cond(tf.random.uniform(shape=[], dtype=tf.float32) < 0.5,
            	    lambda: tf.cond(tf.random.uniform(shape=[], dtype=tf.float32) < 0.7, # 80% erosion, 20% dilation
                        lambda: erode(image, tf.random.uniform(shape=[], minval=1, maxval=6, dtype=tf.int64)), # between 1 and 5 pixels erosion (thicker)
                        lambda: dilate(image, tf.random.uniform(shape=[], minval=1, maxval=2, dtype=tf.int64))), # either 1 or 2 pixels thinner
                    lambda: image)
    return (image, boxes)

# for eagerly testing the augmentation on *.tfrecord
def test(path: str, num_samples: int):
    dataset = tf.data.TFRecordDataset(path)

    ft_desc = {
        'image/height': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'image/width': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'image/filename': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'image/source_id': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'image/encoded': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'image/format': tf.io.FixedLenFeature([], tf.string, default_value=b'jpeg'),
        'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
        'image/object/class/text': tf.io.VarLenFeature(tf.string),
        'image/object/class/label': tf.io.VarLenFeature(tf.int64)}

    dataset = dataset.map(lambda example: tf.io.parse_single_example(example, ft_desc))

    def augment_dataset(example):
        boxes = tf.stack([example['image/object/bbox/ymin'].values, example['image/object/bbox/xmin'].values, example['image/object/bbox/ymax'].values, example['image/object/bbox/xmax'].values], axis=1)
        img = tf.cast(tf.io.decode_jpeg(example['image/encoded']), dtype=tf.float32)

        img, boxes = augment(img, boxes)

        return img, boxes

    dataset = dataset.map(augment_dataset)

    for img, boxes in dataset.take(num_samples):
        img = img.numpy().astype(np.uint8)

        for box in boxes.numpy():
            xmin = box[1] * img.shape[1]
            ymin = box[0] * img.shape[0]
            xmax = box[3] * img.shape[1]
            ymax = box[2] * img.shape[0]
        
            xmin = int(xmin)
            ymin = int(ymin)
            xmax = int(xmax)
            ymax = int(ymax)

            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), thickness=3)

        cv2.imshow('', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    test('./ObjectDetection/data/val.tfrecord', 5)