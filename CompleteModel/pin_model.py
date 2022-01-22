from turtle import pos
from typing import Tuple
import tensorflow as tf
import numpy as np
import sleap

from img_bbox_processing import erode
import helper

class PinDetectionModel(tf.Module):
    def __init__(self, saved_model_path, num_classes=42):
        super().__init__(name='PinDetectionModel')
        self._model = tf.keras.models.load_model(saved_model_path)
        self._img_size: Tuple[int, int] = [input.type_spec.shape[1:3] for input in self._model.inputs if input.name=='input1'][0]
        self._num_classes = num_classes
        assert [input.type_spec.shape[1] for input in self._model.inputs if input.name=='input2'][0] == self._num_classes

    @tf.function(input_signature=[tf.TensorSpec((None, None, None, 1), dtype=tf.float32), tf.TensorSpec((None, 42), dtype=tf.float32)], experimental_follow_type_hints=True)
    def __call__(self, imges: tf.Tensor, class_proposals: tf.Tensor):
        imges = tf.image.resize(imges, self._img_size)
        heatmaps = self._model({"input1": imges, "input2": class_proposals})

        # get local maxima (order x,y)
        peak_pos, peak_vals, batch_ind, _ = sleap.nn.peak_finding.find_local_peaks(heatmaps, threshold=0.2, refinement='integral')
        peak_pos = peak_pos / tf.cast(tf.shape(heatmaps)[1], tf.float32)

        classes_arr = tf.TensorArray(tf.int32, size=0, dynamic_size=True, element_shape=tf.TensorShape((None)))
        batch_ids_arr = tf.TensorArray(tf.int32, size=0, dynamic_size=True, element_shape=tf.TensorShape((None)))
        pins_arr = tf.TensorArray(tf.float32, size=0, dynamic_size=True, element_shape=tf.TensorShape((2)))
        pin_cmp_ids = tf.TensorArray(tf.int32, size=0, dynamic_size=True, element_shape=tf.TensorShape((None)))

        for i in tf.range(tf.shape(imges)[0]):
            if tf.size(tf.where(batch_ind == i)) > 0:
                pins = tf.gather_nd(peak_pos, tf.where(batch_ind == i))
                pin_vals = tf.gather_nd(peak_vals, tf.where(batch_ind == i))

                class_id = tf.argmax(class_proposals[i], output_type=tf.int32)
                pins = self._assert_correct_pin_count(pins, pin_vals, class_id)

                if tf.size(pins) != 0:                
                    classes_arr = classes_arr.write(classes_arr.size(), class_id)
                    batch_ids_arr = batch_ids_arr.write(batch_ids_arr.size(), i)

                    for pin in pins:
                        pins_arr = pins_arr.write(pins_arr.size(), pin)
                        pin_cmp_ids = pin_cmp_ids.write(pin_cmp_ids.size(), classes_arr.size() - 1)

        return {'classes': classes_arr.stack(), 'sample_ind': batch_ids_arr.stack(), 'pins': pins_arr.stack(), 'pin_cmp_ids': pin_cmp_ids.stack()}
    
    OPV = 27
    S3 = 34
    MIC = 25
    SPK = 35
    POT = 30
    ID_TO_NUM_PINS = tf.constant([  2, 2, 2, 2, 2,
                        2, 2, 2, 2, 2,
                        1, 1, 1, 2, 2,
                        3, 3, 2, 2, 2,
                        2, 3, 3, 3, 3,
                        2, 3, 3, 1, 3,
                        3, 2, 2, 2, 3,
                        2, 2, 2, 2, 2,
                        2, 2], dtype=tf.int32, name='ID_TO_NUM_PINS')

    @staticmethod
    def _assert_correct_opv_s3(pins: tf.Tensor, pin_vals: tf.Tensor, num_pins: tf.Tensor, norm_directions: tf.Tensor, directions: tf.Tensor):
        # cannot reliable reconstruct with 1 pin
        if num_pins == 1:
            return tf.constant([])
        elif num_pins == 2:
            # calculate dot product in order to determine if they are opposite
            if tf.tensordot(norm_directions[0], norm_directions[1], 1) > 0:
                # same side
                new_pin = tf.constant([0.5, 0.5], tf.float32) - helper.normalize_vector(directions[0] + directions[1]) * (helper.vector_length(directions[0]) + helper.vector_length(directions[1])) / 2.0
                return tf.concat([pins, tf.expand_dims(new_pin, 0)], axis=0)
            else:
                # opposite -> cannot reliable reconstruct
                return tf.constant([])
        else:
            max_score = tf.constant(0.0)
            best_pins = tf.zeros((3, 2), tf.float32)

            for indices in helper.k_out_of_n_combinations(3, num_pins):
                # check all 3 possibilities for the single pin -> choose best for score
                min_pos_error = tf.constant(tf.float32.max, tf.float32)
                for i in tf.range(3):
                    a = indices[tf.math.floormod(i + 1, 3)]
                    b = indices[tf.math.floormod(i + 2, 3)]

                    pos_error = helper.vector_length(directions[indices[i]] + helper.normalize_vector(directions[a] + directions[b]) * (helper.vector_length(directions[a]) + helper.vector_length(directions[b])) / 2.0)
                    pos_error = tf.squeeze(pos_error)

                    if pos_error < min_pos_error:
                        min_pos_error = pos_error

                score = tf.reduce_mean(tf.gather(pin_vals, indices)) - min_pos_error

                if score > max_score:
                    best_pins = tf.ensure_shape(tf.gather_nd(pins, tf.expand_dims(indices, -1)), (3,2))
                    max_score = score

            return best_pins

    @staticmethod
    def _assert_correct_pin_count(pins: tf.Tensor, pin_vals: tf.Tensor, class_id: tf.Tensor):
        num_pins = tf.shape(pins)[0]

        if num_pins == 0:
            return tf.constant([])

        directions = pins - tf.constant([0.5, 0.5], tf.float32)    # vectors from center to pin
        norm_directions = helper.normalize_vector(directions)

        if num_pins == PinDetectionModel.ID_TO_NUM_PINS[class_id]:
            return pins
            
        if class_id == PinDetectionModel.OPV or class_id == PinDetectionModel.S3:
            return PinDetectionModel._assert_correct_opv_s3(pins, pin_vals, num_pins, norm_directions, directions)
        return pins