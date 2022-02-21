import os
from typing import List

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
        texts = example['image/object/class/text'].values

        return img, boxes, texts

    dataset = dataset.map(lambda example: tf.io.parse_single_example(example, ft_desc)).map(extract_img_boxes)

    aspect_ratios: List[float] = []
    heights: List[int] = []
    widths: List[int] = []
    num_boxes = []

    for img, boxes, texts in dataset:
        img = img.numpy().astype(np.uint8)
        sf = 640.0 / np.amax(img.shape)

        num_boxes.append(boxes.shape[0])

        for box, text in zip(boxes.numpy(), texts.numpy()):
            xmin = box[1] * img.shape[1] * sf
            ymin = box[0] * img.shape[0] * sf
            xmax = box[3] * img.shape[1] * sf
            ymax = box[2] * img.shape[0] * sf
        
            xmin = int(xmin)
            ymin = int(ymin)
            xmax = int(xmax)
            ymax = int(ymax)

            ratio = (xmax - xmin) / (ymax - ymin)
            height = ymax - ymin
            width = xmax - xmin

            if ratio < 1.0:
                ratio = 1.0 / ratio

            aspect_ratios.append(ratio)
            heights.append(height)
            widths.append(width)

    return aspect_ratios, heights, widths, num_boxes

if __name__ == '__main__':
    aspect_ratios = []
    heights = []
    widths = []
    num_boxes = []

    for file in os.listdir('./ObjectDetection/data'):
        file = os.path.join('./ObjectDetection/data', file)
        if os.path.isfile(file) and os.path.splitext(file)[1] == '.tfrecord':
            ratios, h, w, n_b = analyze(file)
            aspect_ratios.extend(ratios)
            heights.extend(h)
            widths.extend(w)
            num_boxes.extend(n_b)

    aspect_ratios = np.array(aspect_ratios)
    aspect_ratios = aspect_ratios[aspect_ratios < np.percentile(aspect_ratios, 95)]

    heights = np.array(heights)
    heights = heights[heights < np.percentile(heights, 95)]

    widths = np.array(widths)
    widths = widths[widths < np.percentile(widths, 95)]

    fig, axs = plt.subplots(4)
    fig.tight_layout(pad=2.0)

    axs[0].set_title('aspect ratios')
    axs[0].hist(aspect_ratios, bins='auto')

    axs[1].set_title('heights')
    axs[1].hist(heights, bins='auto')

    axs[2].set_title('widths')
    axs[2].hist(widths, bins='auto')

    axs[3].set_title('num boxes')
    axs[3].hist(num_boxes, bins='auto')

    plt.savefig('./ObjectDetection/DataAnalysis/plot.png')