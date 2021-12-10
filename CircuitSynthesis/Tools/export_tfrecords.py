import os
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from typing import Dict, List, Tuple
import cv2
import random

from .dataset_utils import *
from .autoroute import RoutedCircuit
from .render import draw_routed_circuit
from .split_circuits import split_circuit

physical_devices = tf.config.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("No GPU detected!")

def _parse_fine_to_coarse(path: str = './DataProcessing/ObjectDetection/fine_to_coarse_labels.txt'):
    convert_dict = {}
    with open(path, 'r') as f:
        for l in f:
            values = l.split(',')
            convert_dict[values[0]] = (values[1], int(values[2]))
    
    return convert_dict

def export_label_map(dest_path, src_path: str = 'ObjectDetection/fine_to_coarse_labels.txt'):
    fine_to_coarse = _parse_fine_to_coarse(src_path)

    with open(dest_path, 'w') as f:
        for name, id in set(fine_to_coarse.values()):
            f.write(f'item {{\n\tid: {int(id)}\n\tname: "{name}"\n}}\n')

def _circuit_to_examples(circ: RoutedCircuit, label_convert: Dict[str, Tuple[str, int]]):
    img = draw_routed_circuit(circ, labels=True)

    bboxs = [(cmp.pos[0], cmp.pos[1], cmp.cmp.component_img.shape[1] + cmp.pos[0], cmp.cmp.component_img.shape[0]+ cmp.pos[1]) for cmp in circ.components]
    
    tf_label_and_data = []

    for new_bboxs, indices, img in split_circuit(bboxs, img):
        encoded_image = tf.io.encode_jpeg(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)).numpy()
        img_h, img_w = img.shape

        xmins = []
        ymins = []
        ymaxs = []
        xmaxs = []
        types = []
        ids = []

        for bbox, i in zip(new_bboxs, indices):
            types.append(label_convert[circ.components[i].type_id][0].encode('utf8'))
            ids.append(label_convert[circ.components[i].type_id][1])
            xmins.append(bbox[0])
            ymins.append(bbox[1])
            xmaxs.append(bbox[2])
            ymaxs.append(bbox[3])

        tf_label_and_data.append(tf.train.Example(features=tf.train.Features(feature={
            'image/height': int64_feature(img_h),
            'image/width': int64_feature(img_w),
            'image/filename': bytes_feature(b''),
            'image/source_id': bytes_feature(b''),
            'image/encoded': bytes_feature(encoded_image),
            'image/format': bytes_feature(b'jpeg'),
            'image/object/bbox/xmin': float_list_feature(xmins),
            'image/object/bbox/xmax': float_list_feature(xmaxs),
            'image/object/bbox/ymin': float_list_feature(ymins),
            'image/object/bbox/ymax': float_list_feature(ymaxs),
            'image/object/class/text': bytes_list_feature(types),
            'image/object/class/label': int64_list_feature(ids),
        })))

    return tf_label_and_data

def export_circuits(circuits: List[RoutedCircuit], train_path, val_path, fine_coarse_path = './DataProcessingObjectDetection/fine_to_coarse_labels.txt', val_split: float = 0.2):
    '''Export a list of RoutedCircuit to a train and validation TFRecord file usable by the tensorflow object detection API.'''
    random.shuffle(circuits)

    label_convert = _parse_fine_to_coarse(fine_coarse_path)

    num_train = int(len(circuits) * (1 - val_split))

    with tf.io.TFRecordWriter(train_path) as train_writer:
        for circ in circuits[:num_train]:
            for example in _circuit_to_examples(circ, label_convert):
                train_writer.write(example.SerializeToString())
    
    if num_train < len(circuits):
        with tf.io.TFRecordWriter(val_path) as val_writer:
            for circ in circuits[num_train:]:
                for example in _circuit_to_examples(circ, label_convert):
                    val_writer.write(example.SerializeToString())

def inspect_record(path, num):
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

    for example in dataset.take(num):
        img = tf.io.decode_jpeg(example['image/encoded']).numpy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        objects = zip(example['image/object/bbox/xmin'].values.numpy(), example['image/object/bbox/xmax'].values.numpy(), example['image/object/bbox/ymin'].values.numpy(), example['image/object/bbox/ymax'].values.numpy(), example['image/object/class/text'].values.numpy(), example['image/object/class/label'].values.numpy())

        for xmin, xmax, ymin, ymax, text, label in objects:
            xmin *= img.shape[1]
            ymin *= img.shape[0]
            xmax *= img.shape[1]
            ymax *= img.shape[0]
        
            xmin = int(xmin)
            ymin = int(ymin)
            xmax = int(xmax)
            ymax = int(ymax)

            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (int(label / 16 * 255), 255, 255), thickness=3)

        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

        cv2.imshow('', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()