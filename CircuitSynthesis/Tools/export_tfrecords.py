import os
import typing
import numpy

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from typing import List
import cv2
import random

from .dataset_util import *
from .autoroute import RoutedCircuit
from .render import draw_routed_circuit

physical_devices = tf.config.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("No GPU detected!")

def _circuit_to_example(circ: RoutedCircuit):
    img = draw_routed_circuit(circ, labels=True)
    img_h, img_w = img.shape
    encoded_image = cv2.imencode('.jpg', img)[1].tostring()

    xmins = []
    ymins = []
    heights = []
    widths = []
    types = []

    for cmp in circ.components:
        types.append(cmp.type_id.encode('utf8'))
        xmins.append(cmp.pos[0] / img_w)
        ymins.append(cmp.pos[1] / img_h)
        widths.append(cmp.cmp.component_img.shape[1] / img_w)
        heights.append(cmp.cmp.component_img.shape[0] / img_h)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(img_h),
        'image/width': int64_feature(img_w),
        'image/encoded': bytes_feature(encoded_image),
        'object/bbox/xmins': float_list_feature(xmins),
        'object/bbox/ymins': float_list_feature(ymins),
        'object/bbox/heights': float_list_feature(heights),
        'object/bbox/widths': float_list_feature(widths),
        'object/types': bytes_list_feature(types)
    }))

    return tf_example

def export_circuits(circuits: List[RoutedCircuit], train_path, val_path, val_split: float = 0.2):
    random.shuffle(circuits)

    num_train = int(len(circuits) * (1 - val_split))

    with tf.io.TFRecordWriter(train_path) as train_writer:
        for circ in circuits[:num_train]:
            train_writer.write(_circuit_to_example(circ).SerializeToString())
    
    with tf.io.TFRecordWriter(val_path) as val_writer:
        for circ in circuits[num_train:]:
            val_writer.write(_circuit_to_example(circ).SerializeToString())

def inspect_record(path):
    dataset = tf.data.TFRecordDataset(path)
    
    ft_desc = {
        'image/height': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'image/width': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'image/encoded': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'object/bbox/xmins': tf.io.VarLenFeature(tf.float32),
        'object/bbox/ymins': tf.io.VarLenFeature(tf.float32),
        'object/bbox/heights': tf.io.VarLenFeature(tf.float32),
        'object/bbox/widths': tf.io.VarLenFeature(tf.float32),
        'object/types': tf.io.VarLenFeature(tf.string)
    }

    @tf.function
    def parse_example(example_proto):
        return tf.io.parse_single_example(example_proto, ft_desc)

    dataset = dataset.map(map_func=parse_example)

    for entry in dataset.take(-1):
        img = cv2.imdecode(tf.io.decode_raw(entry['image/encoded'], tf.uint8).numpy(), cv2.IMREAD_UNCHANGED)
        cv2.imshow('', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()