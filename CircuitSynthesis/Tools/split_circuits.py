from typing import List, Tuple

import numpy as np

MAX_SUB_SIZE = 800
SUB_SIZE = 640
OVERLAP = 50
MIN_COMPONENT_AREA_REL = 0.3

def split_circuit(bboxs: List[Tuple[float, float, float, float]], img: np.ndarray):
    '''
    bboxs: list of (xmin, ymin, xmax, ymax)
    returned bboxs are in [0; 1]
    '''

    # size is [width, height]
    size = np.array(img.shape)[1::-1]

    if np.all(size < SUB_SIZE):
        img_h, img_w = img.shape[:2]
        return [([(bbox[0] / img_w,
                    bbox[1] / img_h,
                    bbox[2] / img_w,
                    bbox[3] / img_h) for bbox in bboxs], list(range(len(bboxs))), img, [1.0] * len(bboxs))]
    
    num = np.maximum(np.ceil((size - OVERLAP) / (MAX_SUB_SIZE - OVERLAP)), 1).astype(np.uint8)
    sub_size = ((size + (num - 1) * OVERLAP) / num).astype(np.int32)
    stride = sub_size - OVERLAP

    result: List[Tuple[List[Tuple[float,float, float, float]], List[int], np.ndarray]] = []
    
    for i_x in range(num[0]):
        for i_y in range(num[1]):
            offset = stride * np.array([i_y, i_x])

            new_img = img[offset[0]:offset[0]+sub_size[1], offset[1]:offset[1]+sub_size[0]]

            img_h, img_w = new_img.shape[:2]
            new_bboxs = []
            indices = []
            weights = []

            for i, bbox in enumerate(bboxs):
                new_box = (np.clip((bbox[0] - offset[1]) / img_w, 0, 1),
                    np.clip((bbox[1] - offset[0]) / img_h, 0, 1),
                    np.clip((bbox[2] - offset[1]) / img_w, 0, 1),
                    np.clip((bbox[3] - offset[0]) / img_h, 0, 1))

                inter_area = (new_box[2] - new_box[0]) * (new_box[3] - new_box[1])
                tot_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) / img_h / img_w

                if inter_area > tot_area * MIN_COMPONENT_AREA_REL:
                    new_bboxs.append(new_box)
                    indices.append(i)
                    weights.append(max(((inter_area / tot_area) - MIN_COMPONENT_AREA_REL) / (1.0 - MIN_COMPONENT_AREA_REL), 0.0))
                    
            result.append((new_bboxs, indices, new_img, weights))

    return result