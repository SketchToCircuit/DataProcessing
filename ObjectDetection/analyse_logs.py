import os
os.environ['CUDA_VSIBLE_DEVICES'] = '-1'
import tensorflow as tf
from tensorflow.python.summary.summary_iterator import summary_iterator
import struct

MODEL_PATH = './ObjectDetection/models/ssd_resnet101_640'

total_time = 0
for version_folder in os.listdir(MODEL_PATH):
    if os.path.isdir(os.path.join(MODEL_PATH, version_folder)):
        path = os.path.join(MODEL_PATH, version_folder, 'train')

        avg_steps_per_second = 0.0
        i = 0
        max_step = 0

        for file in os.listdir(path):
            file = os.path.join(path, file)
            if 'events.out.tfevents' in file:
                for event in summary_iterator(file):
                    for value in event.summary.value:
                        if value.tag == 'steps_per_sec':
                            i += 1
                            avg_steps_per_second += struct.unpack('f', value.tensor.tensor_content)[0]
                    if event.step > max_step:
                        max_step = event.step
        avg_steps_per_second /= i
        train_time = max_step / avg_steps_per_second
        total_time += train_time
print(f'{total_time / 60 / 60}h')