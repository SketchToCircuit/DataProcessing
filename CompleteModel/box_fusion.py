from typing import Tuple, List
import numpy as np
import tensorflow as tf

from helper import remove_idx_from_tensor, remove_indices_from_tensor, iou_overlap_coeff

class FuseBoxes(tf.Module):
    NUM_COMBINED_CLASSES = 17

    def __init__(self, hyperparameters=None):
        super().__init__(name='FuseBoxes')
        self._other_obj_mask, self._combined_object_masks = self._get_combined_object_masks()

        if hyperparameters is None:
            hyperparameters = {
                'box_final_thresh': 0.6,
                'box_overlap_thresh': 0.4,
                'box_iou_weight': 0.4,
                'box_weighting_overlap': 0.7,
                'box_certainty_cluster_count': 0.6,
                'box_certainty_combined_scores': 0.7
            }

        self._hyperparameters = hyperparameters

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, 4), dtype=tf.int32), tf.TensorSpec(shape=(None,42), dtype=tf.float32)], experimental_follow_type_hints=True)
    def __call__(self, boxes, scores):
        # choose single object scores
        combined_scores = tf.squeeze(tf.gather(scores, tf.where(self._other_obj_mask == 1), axis=1), axis=-1)

        # combine scores of similar objects
        for mask in self._combined_object_masks:
            masked_scores = tf.squeeze(tf.gather(scores, tf.where(mask == 1), axis=1), axis=-1)
            score = tf.reduce_sum(masked_scores, axis=-1, keepdims=True)
            combined_scores = tf.concat([combined_scores, score], axis=-1)

        combined_scores = tf.ensure_shape(combined_scores, (None, FuseBoxes.NUM_COMBINED_CLASSES))

        boxes, scores, certainty = self._custom_fusion(boxes, combined_scores, scores, overlap_threshold=self._hyperparameters['box_overlap_thresh'])

        final_indices = tf.squeeze(tf.where(certainty > self._hyperparameters['box_final_thresh']), axis=1)

        boxes = tf.gather(boxes, final_indices, axis=0)
        scores = tf.gather(scores, final_indices, axis=0)

        return boxes, scores
  
    def _custom_fusion(self, boxes, combined_scores, original_scores, overlap_threshold=0.5):
        # custom box fusion as a combination of WBF, NMS, and accepts completely enclosed objects as part of a cluster
        resulting_boxes = tf.zeros((0,4), tf.int32)
        resulting_scores = tf.zeros((0,42), tf.float32)
        certainties = tf.zeros((0,), tf.float32)

        num_boxes = tf.shape(boxes)[0]

        # stop if no more unorderd boxes exist
        while num_boxes > 0:
            tf.autograph.experimental.set_loop_options(shape_invariants=[
                (resulting_boxes, tf.TensorShape((None, 4))),
                (resulting_scores, tf.TensorShape((None, 42))),
                (certainties, tf.TensorShape((None, )))])

            # start with object with higest score
            box_id = tf.argmax(tf.reduce_sum(combined_scores, axis=1))

            box = boxes[box_id]
            combined_score = combined_scores[box_id]
            combined_class = tf.argmax(combined_score)
            original_score = original_scores[box_id]

            # IoU and overlap coeff.
            iou, oc = iou_overlap_coeff(boxes, box)
            overlap = iou * self._hyperparameters['box_iou_weight'] + oc * (1.0 - self._hyperparameters['box_iou_weight'])

            # mask for objects with the same (combined) class
            same_class = tf.where(tf.argmax(combined_scores, axis=-1) == combined_class, 1.0, 0.0)

            # indices for this cluster
            cluster_indices = tf.squeeze(tf.where(overlap * same_class > overlap_threshold), axis=-1)

            # get selected boxes, scores and weights (including starter box)
            selected_boxes = tf.gather(boxes, cluster_indices)
            selected_scores = tf.gather(original_scores, cluster_indices)
            selected_combined_scores = tf.gather(combined_scores, cluster_indices)

            # calculate weights (prefer small boxes with high overlap, and high scores)
            score_overlap_w = tf.gather(overlap * combined_scores[:, combined_class], cluster_indices)
            size_scores = 1.0 / tf.cast((selected_boxes[:, 2] - selected_boxes[:, 0]) * (selected_boxes[:, 3] - selected_boxes[:, 1]), tf.float32)
            weights =   score_overlap_w / tf.reduce_sum(score_overlap_w) * self._hyperparameters['box_weighting_overlap'] + \
                        size_scores / tf.reduce_sum(size_scores) * ( 1.0 - self._hyperparameters['box_weighting_overlap'])

            # calculate weighted boxes and scores
            avg_box = tf.cast(tf.reduce_sum(tf.cast(selected_boxes, tf.float32) * weights[:, None], axis=0), tf.int32)
            avg_scores = tf.reduce_sum(selected_scores * weights[:, None], axis=0)
            avg_combined_scores = tf.reduce_sum(selected_combined_scores * weights[:, None], axis=0)

            # calculate certainty for this cluster
            certainty = (1.0 - 1.0 / tf.cast((tf.size(cluster_indices) + 1), tf.float32)) * self._hyperparameters['box_certainty_cluster_count'] + \
                        (avg_combined_scores[combined_class] * self._hyperparameters['box_certainty_combined_scores'] + \
                        tf.reduce_max(avg_scores) * (1.0 - self._hyperparameters['box_certainty_combined_scores'])) * (1.0 - self._hyperparameters['box_certainty_cluster_count'])

            # append these boxes and scores to result
            resulting_boxes = tf.concat([resulting_boxes, avg_box[None, :]], axis=0)
            certainties = tf.concat([certainties, tf.expand_dims(certainty, 0)], axis=0)
            resulting_scores = tf.concat([resulting_scores, avg_scores[None, :]], axis=0)

            # remove cluster from boxes
            boxes = remove_indices_from_tensor(boxes, cluster_indices)
            combined_scores = remove_indices_from_tensor(combined_scores, cluster_indices)
            original_scores = remove_indices_from_tensor(original_scores, cluster_indices)
            num_boxes = num_boxes - tf.size(cluster_indices)
            
        return resulting_boxes, resulting_scores, certainties

    def _get_combined_object_masks(self) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        combined_object_masks = []

        circular_obj_mask = np.zeros(42)
        circular_obj_mask[np.array([0, 13, 14, 19, 20, 39, 37, 37, 40])] = 1
        combined_object_masks.append(tf.convert_to_tensor(circular_obj_mask, dtype=tf.int32))

        diode_obj_mask = np.zeros(42)
        diode_obj_mask[np.array([6, 7, 8, 18])] = 1
        combined_object_masks.append(tf.convert_to_tensor(diode_obj_mask, dtype=tf.int32))

        btn_obj_mask = np.zeros(42)
        btn_obj_mask[np.array([2, 3, 32, 33, 34, 28])] = 1
        combined_object_masks.append(tf.convert_to_tensor(btn_obj_mask, dtype=tf.int32))

        mfet_obj_mask = np.zeros(42)
        mfet_obj_mask[np.array([21, 22, 23, 24])] = 1
        combined_object_masks.append(tf.convert_to_tensor(mfet_obj_mask, dtype=tf.int32))

        jfet_obj_mask = np.zeros(42)
        jfet_obj_mask[np.array([15, 16])] = 1
        combined_object_masks.append(tf.convert_to_tensor(jfet_obj_mask, dtype=tf.int32))

        bi_obj_mask = np.zeros(42)
        bi_obj_mask[np.array([28, 29])] = 1
        combined_object_masks.append(tf.convert_to_tensor(bi_obj_mask, dtype=tf.int32))

        r_obj_mask = np.zeros(42)
        r_obj_mask[np.array([30, 31, 17, 9])] = 1
        combined_object_masks.append(tf.convert_to_tensor(r_obj_mask, dtype=tf.int32))

        bat_obj_mask = np.zeros(42)
        bat_obj_mask[np.array([1, 38])] = 1
        combined_object_masks.append(tf.convert_to_tensor(bat_obj_mask, dtype=tf.int32))

        gnd_obj_mask = np.zeros(42)
        gnd_obj_mask[np.array([10, 12])] = 1
        combined_object_masks.append(tf.convert_to_tensor(gnd_obj_mask, dtype=tf.int32))

        c_obj_mask = np.zeros(42)
        c_obj_mask[np.array([4, 5])] = 1
        combined_object_masks.append(tf.convert_to_tensor(c_obj_mask, dtype=tf.int32))

        other_obj_mask = tf.zeros((42), dtype=tf.int32)
        for mask in combined_object_masks:
            other_obj_mask = other_obj_mask + mask
        other_obj_mask = 1 - other_obj_mask

        return other_obj_mask, combined_object_masks