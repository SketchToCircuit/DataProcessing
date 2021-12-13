import os
from typing import Set

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import tensorflow_addons as tfa
import cv2
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import os

def analyze(path: str):
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

    def extract_img_boxes(example):
        boxes = tf.stack([example['image/object/bbox/ymin'].values, example['image/object/bbox/xmin'].values, example['image/object/bbox/ymax'].values, example['image/object/bbox/xmax'].values], axis=1)
        img = tf.cast(tf.io.decode_jpeg(example['image/encoded']), dtype=tf.float32)

        return img, boxes

    dataset = dataset.map(lambda example: tf.io.parse_single_example(example, ft_desc)).map(extract_img_boxes)

    aspect_ratios: Set[float] = set()

    for img, boxes in dataset.take(5):
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

            ratio = (xmax - xmin) / (ymax - ymin)

            if ratio < 1.0:
                ratio = 1.0 / ratio

            aspect_ratios.add(ratio)
    
    return aspect_ratios

if __name__ == '__main__':
    aspect_ratios: Set[float] = set()

    for file in os.listdir('./ObjectDetection/data'):
        file = os.path.join('./ObjectDetection/data', file)
        if os.path.isfile(file) and os.path.splitext(file)[1] == '.tfrecord':
            aspect_ratios.update(analyze(file))

    plt.hist(aspect_ratios, bins=math.ceil(math.sqrt(len(aspect_ratios))))
    plt.savefig('./ObjectDetection/DataAnalysis/plot.png')