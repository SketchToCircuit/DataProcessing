import random
import numpy as np
import math
import cv2
import scipy.signal
import tensorflow as tf
import Tools.PinDetection.pindetection as pd
from Tools.split_circuits import split_circuit
from Tools.dataset_utils import *
from Tools.squigglylines import Lines

def _getMedianLineThickness(img):
    _, bw = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    dist = cv2.distanceTransform(bw, cv2.DIST_L2, 3)
    maxima = scipy.signal.argrelextrema(dist, np.greater, order=2)
    if maxima[0].size == 0:
        return 3.0
    return max(np.median(dist[maxima]) * 2.0, 1.0)

def _detect_dummy_objects(mask_img):
    obj_mask = np.where(np.any(mask_img < 127, -1), [255], [0]).astype(np.uint8)
    main_pin_mask = np.where(np.logical_and(mask_img[..., 2] < 50, mask_img[..., 0] > 150), [255], [0]).astype(np.uint8)
    special_pin_mask = np.where(np.logical_and(np.logical_and(mask_img[..., 1] > 20, mask_img[..., 2] < 50), mask_img[..., 0] < 50), [255], [0]).astype(np.uint8)

    # cv2.imshow('o', obj_mask)
    # cv2.imshow('m', main_pin_mask)
    # cv2.imshow('s', special_pin_mask)
    # cv2.waitKey()

    main_pin_contours, _ = cv2.findContours(main_pin_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1)
    special_pin_contours, _ = cv2.findContours(special_pin_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1)

    main_pins = []
    for cnt in main_pin_contours:
        try:
            m = cv2.moments(cnt)
            x = m['m10'] / m['m00']
            y = m['m01'] / m['m00']
            main_pins.append(np.array([x, y]))
        except Exception:
            print('empty contour')
            pass
    
    special_pins = []
    for cnt in special_pin_contours:
        try:
            m = cv2.moments(cnt)
            x = m['m10'] / m['m00']
            y = m['m01'] / m['m00']
            special_pins.append(np.array([x, y]))
        except Exception:
            print('empty contour')
            pass
    
    obj_contours, _ = cv2.findContours(obj_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1)
    
    objects = []
    
    for obj_cnt in obj_contours:
        try:
            m = cv2.moments(obj_cnt)
            x = m['m10'] / m['m00']
            y = m['m01'] / m['m00']
        except Exception:
            print('empty contour')
            pass

        current_pins = []
        current_special_pins = []
        special_pin_types = []
    
        for pin in main_pins:
            if cv2.pointPolygonTest(obj_cnt, pin, measureDist=False) > 0:
                current_pins.append(pin)
    
        for pin in special_pins:
            if cv2.pointPolygonTest(obj_cnt, pin, measureDist=False) > 0:
                current_special_pins.append(pin)
                special_pin_types.append(round(mask_img[int(pin[1]), int(pin[0]), 1] / 85) - 1)
    
        objects.append((np.array([x, y]), current_pins, current_special_pins, special_pin_types))
    
    return objects

def _overlay_patch(patch, img, offset):
    offset = offset[::-1]

    patch_start_indx = np.maximum(-offset, 0)
    patch_end_indx = np.minimum(np.array(patch.shape) + offset, np.array(img.shape)) - offset
    patch_size = patch_end_indx - patch_start_indx

    if (np.any(patch_size <= 0)):
        return

    offset = np.maximum(offset, 0)

    img[offset[0]:offset[0]+patch_size[0], offset[1]:offset[1]+patch_size[1]] = img[offset[0]:offset[0]+patch_size[0], offset[1]:offset[1]+patch_size[1]] / 255.0 * patch[patch_start_indx[0]:patch_end_indx[0], patch_start_indx[1]:patch_end_indx[1]]

def _overlay_component(cmp: pd.Component, img, offset):
    _overlay_patch(cmp.component_img, img, offset)
    if random.random() < 0.2:
        _overlay_patch(cmp.label_img, img, (offset + cmp.label_offset).astype(int))

def _place_single_pinned(raw_components, pin, center, line_thickness, img):
    GNDS = ['GND', 'GND_F', 'GND_C']
    type = 'PIN' if random.random() > 0.5 else random.choice(GNDS)

    cmp: pd.Component = random.choice(raw_components[type]).load()
    cmp.component_img = cv2.erode(cmp.component_img, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    cmp.label_img = cv2.erode(cmp.label_img, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    cmp.scale(line_thickness / _getMedianLineThickness(cmp.component_img))

    orientation_vec = pin - center

    if abs(orientation_vec[0]) > abs(orientation_vec[1]):
        if orientation_vec[0] > 0:
            # pin on the right
            if cmp.type in GNDS:
                cmp.rotate(-90)
            else:
                cmp.rotate(180)
        else:
            # pin on the left
            if cmp.type in GNDS:
                cmp.rotate(90)
    else:
        if orientation_vec[1] > 0:
            # pin on bottom
            if cmp.type in GNDS:
                cmp.rotate(180)
            else:
                cmp.rotate(90)
        else:
            #pin on top
            if cmp.type == 'PIN':
                cmp.rotate(-90)

    offset = np.round(pin - list(cmp.pins.values())[0].position).astype(int)

    _overlay_component(cmp, img, offset)

    return (offset[0], offset[1], offset[0] + cmp.component_img.shape[1], offset[1] + cmp.component_img.shape[0]), type

def _place_mic_spk(raw_components, pins, center, img):
    type = random.choice(['MIC', 'SPK'])
    cmp: pd.Component = random.choice(raw_components[type]).load()
    cmp_pins = [*cmp.pins.values()]

    if cmp_pins[0].position[0] > cmp_pins[1].position[0]:
        smaller_cmp_pin = 0
    else:
        smaller_cmp_pin = 1

    orientation = (pins[0] + pins[1]) / 2.0 - center

    if abs(orientation[0]) > abs(orientation[1]):
        sf = abs((pins[0] - pins[1])[1]) / abs((cmp_pins[0].position - cmp_pins[1].position)[1])
        cmp.scale(sf)

        if orientation[0] > 0:
            # pins on the right
            cmp.rotate(180)

            if pins[0][0] > pins[1][0]:
                smaller_pin = 0
            else:
                smaller_pin = 1
        else:
            # pins on the left
            if pins[0][0] > pins[1][0]:
                smaller_pin = 1
            else:
                smaller_pin = 0
        
        larger_pin = (smaller_pin + 1) % 2
        larger_cmp_pin = (smaller_cmp_pin + 1) % 2

        cmp_pins = [*cmp.pins.values()]

        if np.sign(pins[larger_pin][1] - pins[smaller_pin][1]) != np.sign(cmp_pins[larger_cmp_pin].position[1] - cmp_pins[smaller_cmp_pin].position[1]):
            cmp.flip(vert=True)
    else:
        sf = abs((pins[0] - pins[1])[0]) / abs((cmp_pins[0].position - cmp_pins[1].position)[1])
        cmp.scale(sf)

        if orientation[1] > 0:
            # pins on bottom
            cmp.rotate(90)

            if pins[0][1] > pins[1][1]:
                smaller_pin = 0
            else:
                smaller_pin = 1
        else:
            # pins on top
            cmp.rotate(-90)
        
            if pins[0][1] > pins[1][1]:
                smaller_pin = 1
            else:
                smaller_pin = 0

        larger_pin = (smaller_pin + 1) % 2
        larger_cmp_pin = (smaller_cmp_pin + 1) % 2

        cmp_pins = [*cmp.pins.values()]

        if np.sign(pins[larger_pin][0] - pins[smaller_pin][0]) != np.sign(cmp_pins[larger_cmp_pin].position[0] - cmp_pins[smaller_cmp_pin].position[0]):
            cmp.flip(hor=True)
    
    cmp_pins = [*cmp.pins.values()]
    offset = pins[smaller_pin] - cmp_pins[smaller_cmp_pin].position
    _overlay_component(cmp, img, offset.astype(int))

    return (offset[0], offset[1], offset[0] + cmp.component_img.shape[1], offset[1] + cmp.component_img.shape[0]), type

def _place_double_pinned(raw_components, pins, center, line_thickness, img):
    # 10% probability to only draw a line
    if random.random() < 0.1:
        Lines.squigglyline(pins[0][0], pins[0][1], pins[1][0], pins[1][1], img, int(math.ceil(line_thickness)), 0)
        return None, None

    vec_a = pins[0] - center
    vec_b = pins[1] - center
    a = np.dot(vec_a, vec_b) / (math.sqrt(np.sum(np.square(vec_a))) * math.sqrt(np.sum(np.square(vec_b))))

    if a > -0.707:
        return _place_mic_spk(raw_components, pins, center, img)
    else:
        both_versions = {'A_H', 'A_V', 'U_AC_H', 'U_AC_V', 'V_H', 'V_V', 'M', 'M_V'}

        two_pins = ['R', 'C', 'L', 'V_V', 'A_V',
        'U1', 'U2', 'I1', 'I2', 'U3', 'BAT', 'U_AC_V', 'M_V',
        'L2', 'LED', 'D', 'S1', 'S2', 'BTN1', 'BTN2', 'V_H',
        'A_H', 'U_AC_H', 'LMP', 'M', 'F', 'D_Z', 'D_S', 'C_P']

        type_weights = [0.5 if t in both_versions else 1.0 for t in two_pins]
        type = random.choices(two_pins, weights=type_weights, k=1)[0]

    cmp: pd.Component = random.choice(raw_components[type]).load()
    cmp.component_img = cv2.erode(cmp.component_img, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    cmp.label_img = cv2.erode(cmp.label_img, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

    cmp_pins = random.sample([*cmp.pins.values()], 2)
    cmp_pin_vec = cmp_pins[1].position - cmp_pins[0].position
    cmp_pin_dist = math.sqrt(np.sum(np.square(cmp_pin_vec)))
    goal_pin_vec = pins[1] - pins[0]
    goal_pin_dist = math.sqrt(np.sum(np.square(goal_pin_vec)))

    cmp.scale(goal_pin_dist / cmp_pin_dist)

    angle = math.atan2(cmp_pin_vec[1], cmp_pin_vec[0]) - math.atan2(goal_pin_vec[1], goal_pin_vec[0])

    cmp.rotate(angle * 180.0 / math.pi)

    offset = pins[0] - cmp_pins[0].position
    _overlay_component(cmp, img, offset.astype(int))

    return (offset[0], offset[1], offset[0] + cmp.component_img.shape[1], offset[1] + cmp.component_img.shape[0]), type

def _place_triple_pinned(raw_components, pins, special_pin, special_pin_type, img):
    special_pin_name = {'OPV': 'out', 'S3': 'out', 'POT': '3',
    'JFET_N': 'gate', 'JFET_P': 'gate', 'MFET_N_D': 'gate', 'MFET_N_E': 'gate', 'MFET_P_D': 'gate', 'MFET_P_E': 'gate',
    'NPN': 'base', 'PNP': 'base'}
    
    if special_pin_type == 0:
        type = random.choice(['OPV', 'S3'])
    elif special_pin_type == 1:
        type = 'POT'
    else:
        type = random.choice(['JFET_N', 'JFET_P', 'MFET_N_D', 'MFET_N_E', 'MFET_P_D', 'MFET_P_E', 'NPN', 'PNP'])

    cmp: pd.Component = random.choice(raw_components[type]).load()

    src_pos = random.sample([pin.position for key, pin in cmp.pins.items() if key != special_pin_name[type]], 2)
    src_pos.append(cmp.pins[special_pin_name[type]].position)
    src_pos = np.array(src_pos, np.float32)

    dst_pos = pins
    dst_pos.append(special_pin)
    dst_pos = np.array(dst_pos, np.float32)
    
    affine_transform = cv2.getAffineTransform(src_pos, dst_pos)

    bbox = np.array([[[0.0, 0.0], [cmp.component_img.shape[1], 0.0], [0.0, cmp.component_img.shape[0]], [cmp.component_img.shape[1], cmp.component_img.shape[0]]]], np.float32)
    bbox = cv2.transform(bbox, affine_transform)[0].astype(int)
    bbox = (np.amin(bbox[:, 0]), np.amin(bbox[:, 1]), np.amax(bbox[:, 0]), np.amax(bbox[:, 1]))

    max_size = max(np.amax(dst_pos[..., 0]) - np.amin(dst_pos[..., 0]), np.amax(dst_pos[..., 1]) - np.amin(dst_pos[..., 1])) * 3.0

    if max(bbox[2] - bbox[0], bbox[3] - bbox[1]) > max_size:
        return None, None

    img[:] = (img * (cv2.warpAffine(cmp.component_img, affine_transform, np.array(img.shape)[1::-1], flags=cv2.BORDER_CONSTANT | cv2.INTER_AREA, borderValue=255) / 255.0)).astype(np.uint8)

    return bbox, type

def place_components(dummy_objects, raw_components, line_thickness, img):
    bboxes = []
    labels = []

    for obj in dummy_objects:
        center, pins, special_pins, special_pin_types = obj

        if len(pins) == 1:
            bbox, label = _place_single_pinned(raw_components, pins[0], center, line_thickness, img)
        elif len(pins) == 2 and len(special_pins) == 0:
            bbox, label = _place_double_pinned(raw_components, pins, center, line_thickness, img)
        elif len(pins) == 2 and len(special_pins) == 1:
            bbox, label = _place_triple_pinned(raw_components, pins, special_pins[0], special_pin_types[0], img)
        else:
            continue

        if bbox is None:
            continue

        bboxes.append(bbox)
        labels.append(label)
    return (bboxes, labels)

def translate_img(img, dx, dy):
    if img.ndim == 2:
        new_img = np.full((img.shape[0] + abs(dy), img.shape[1] + abs(dx)), 255, np.uint8)
    else:
        new_img = np.full((img.shape[0] + abs(dy), img.shape[1] + abs(dx), img.shape[2]), 255, np.uint8)
    new_img[max(0, dy):max(0, dy) + img.shape[0], max(0, dx):max(0, dx) + img.shape[1]] = img

    return new_img

def get_examples(dummy_path, mask_path, raw_components, label_convert):
    dummy_img = cv2.imread(dummy_path, cv2.IMREAD_GRAYSCALE)
    mask_img = cv2.imread(mask_path, cv2.IMREAD_COLOR)

    if dummy_img.shape[:2] != mask_img.shape[:2]:
        return []

    shift_y = random.randint(-int(dummy_img.shape[0] * 0.5), int(dummy_img.shape[0] * 0.5))
    shift_x = random.randint(-int(dummy_img.shape[1] * 0.5), int(dummy_img.shape[1] * 0.5))
    dummy_img = translate_img(dummy_img, shift_x, shift_y)
    mask_img = translate_img(mask_img, shift_x, shift_y)

    line_thickness = _getMedianLineThickness(dummy_img)

    tf_label_and_data = []

    for i in range(4):
        sf = random.uniform(1.5, 3.0) / line_thickness
        img = dummy_img.copy()

        dummy_objects = _detect_dummy_objects(mask_img)

        bboxes, labels = place_components(dummy_objects, raw_components, line_thickness,img)

        img = cv2.resize(img, None, fx=sf, fy=sf, interpolation=cv2.INTER_AREA)
        bboxes = [tuple(a * sf for a in box) for box in bboxes]

        # cv2.imshow('img', img)
        # cv2.waitKey(0)

        for new_bboxs, indices, split_img, weights in split_circuit(bboxes, img):
            encoded_image = tf.io.encode_jpeg(cv2.cvtColor(split_img, cv2.COLOR_GRAY2BGR)).numpy()
            img_h, img_w = split_img.shape

            xmins = []
            ymins = []
            ymaxs = []
            xmaxs = []
            types = []
            ids = []
            adjusted_weights = []

            for bbox, idx, w in zip(new_bboxs, indices, weights):
                types.append(label_convert[labels[idx]][0].encode('utf8'))
                ids.append(label_convert[labels[idx]][1])
                xmins.append(min(max(bbox[0], 0.0), 1.0))
                ymins.append(min(max(bbox[1], 0.0), 1.0))
                xmaxs.append(min(max(bbox[2], 0.0), 1.0))
                ymaxs.append(min(max(bbox[3], 0.0), 1.0))
                adjusted_weights.append(w * 0.7)

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
                'image/object/weight': float_list_feature(adjusted_weights)
            })))
        
        if i < 3:
            dummy_img = cv2.rotate(dummy_img, cv2.ROTATE_90_CLOCKWISE)
            mask_img = cv2.rotate(mask_img, cv2.ROTATE_90_CLOCKWISE)

    return tf_label_and_data

def main():
    from Tools.export_tfrecords import _parse_fine_to_coarse
    raw_components = pd.import_components('./DataProcessing/pindetection_data/data.json')
    label_convert = _parse_fine_to_coarse('./DataProcessing/ObjectDetection/fine_to_coarse_labels.txt')

    for i in range(20, 26):
        print(i)
        get_examples(f'./DataProcessing/CircuitSynthesis/DummySketchData/{i}_dummy.jpg', f'./DataProcessing/CircuitSynthesis/DummySketchData/{i}_mask.jpg', raw_components, label_convert)

if __name__ == '__main__':
    main()