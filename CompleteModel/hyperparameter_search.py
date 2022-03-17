import json
import os
import scipy.stats
import cv2
import math
import base64
import termcolor

import numpy as np
import scipy.signal

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

tf.get_logger().setLevel('ERROR')

tf.config.optimizer.set_experimental_options({
    'layout_optimizer': True,
    'constant_folding': True,
    'shape_optimization': True,
    'remapping': True,
    'arithmetic_optimization': True,
    'dependency_optimization': True,
    'loop_optimization': True,
    'function_optimization': True,
    'scoped_allocator_optimization': True,
    'implementation_selector': True,
    'disable_meta_optimizer': False
})

component_list = ['A', 'BAT', 'BTN1', 'BTN2', 'C', 'C_P', 'D', 'D_S', 'D_Z', 'F', 'GND', 'GND_C', 'GND_F', 'I1', 'I2', 'JFET_N', 'JFET_P', 'L', 'LED', 'LMP', 'M', 'MFET_N_D', 'MFET_N_E', 'MFET_P_D', 'MFET_P_E', 'MIC', 'NPN', 'OPV', 'PIN', 'PNP', 'POT', 'R', 'S1', 'S2', 'S3', 'SPK', 'U1', 'U2', 'U3', 'U_AC', 'V', 'L2']

from combined_model import CombinedModel

def calc_iou(a, b):
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    intersection = [
        max(a[0], b[0]),
        max(a[1], b[1]),
        min(a[2], b[2]),
        min(a[3], b[3]),
    ]
    area_inter = max(intersection[2] - intersection[0], 0) * max(intersection[3] - intersection[1], 0)
    return area_inter / (area_a + area_b - area_inter)

def object_detection_score(classes, boxes, true_components):
    score = 0
    num_undetected = len(true_components)
    already_detected = set()

    for cls, box in zip(classes, boxes):
        selected_cmps = [a for a in enumerate(true_components) if a[1]['component'] == component_list[cls]]

        if len(selected_cmps) == 0:
            score = score - 2
        else:
            max_iou = 0

            for i, cmp in selected_cmps:
                true_box = [cmp['topleft']['y'], cmp['topleft']['x'], cmp['bottomright']['y'], cmp['bottomright']['x']]
                iou = calc_iou(true_box, box)

                if iou > max_iou:
                    max_iou = iou
            
            if max_iou < 0.3:
                score = score - 2
            elif i in already_detected:
                score = score - 1.5
            else:
                score = score + max_iou * 1.5
                num_undetected -= 1
                already_detected.add(i)

    score = score - num_undetected * 3
    
    return score / (len(classes) + 0.5)

def truncated_normal_distribution(mean, std, a, b):
    a_norm, b_norm = (a - mean) / std, (b - mean) / std
    return scipy.stats.truncnorm.rvs(a=a_norm, b=b_norm, loc=mean, scale=std)

def mutate_hyperparameters(param, std):
    mutated = {}
    for k in param.keys():
        mutated[k] = truncated_normal_distribution(param[k], std, 0.0, 1.0)
    return mutated

def main():
    hyperparameters = {
    'pin_peak_thresh': 0.2,
    'pin_val_weight': 0.5,
    'box_final_thresh': 0.6,
    'box_overlap_thresh': 0.7,
    'box_different_class_iou_thresh': 0.8,
    'box_iou_weight': 0.3,
    'box_certainty_cluster_count': 0.8,
    'box_certainty_combined_scores': 0.5
    }

    best_score = -math.inf

    for i in range(1000):
        print(i + 1)

        model = CombinedModel('./ObjectDetection/exported_models/ssd_resnet101_640_v14/saved_model', './PinDetection/exported/1', hyperparameters=hyperparameters, do_not_convert_variables=True)

        summed_score = 0
        for i in range(5):
            img = cv2.imread(f'./CompleteModel/TestData/test{i+1}.jpeg', cv2.IMREAD_GRAYSCALE)
            classes, boxes, pins, pin_cmp_ids = model(base64.urlsafe_b64encode(cv2.imencode('.jpg', img)[1])).values()

            with open(f'./CompleteModel/TestData/test{i+1}.json') as f:
                true_components = json.load(f)

            summed_score += object_detection_score(list(classes.numpy()), list(boxes.numpy()), true_components)

        if summed_score > best_score:
            best_hyperparameters = hyperparameters.copy()
            best_score = summed_score
            print(termcolor.colored('New best hyperparameterse:', 'red'))
            print(f'Score: {summed_score}')
            print(best_hyperparameters)
        
        hyperparameters = mutate_hyperparameters(best_hyperparameters, 1.0 / (i / 20.0 + 1))

main()