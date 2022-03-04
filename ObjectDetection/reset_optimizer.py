from keyword import kwlist
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf

from object_detection import eval_util
from object_detection import inputs
from object_detection import model_lib
from object_detection.builders import optimizer_builder
from object_detection.core import standard_fields as fields
from object_detection.protos import train_pb2
from object_detection.utils import config_util
from object_detection.utils import label_map_util
from object_detection.utils import ops
from object_detection.utils import variables_helper
from object_detection.utils import visualization_utils as vutils

import numpy as np

MODEL_BUILD_UTIL_MAP = model_lib.MODEL_BUILD_UTIL_MAP

SRC_CKPT = './ObjectDetection/models/ssd_resnet101_640/v19/best_ckps/ckpt-318'
DEST_CKPT = './ObjectDetection/models/ssd_resnet101_640/test/ckpt'
PIPELINE_CONFIG_PATH = './ObjectDetection/models/ssd_resnet101_640/v19/pipeline.config'
MODEL_DIR = './ObjectDetection/models/ssd_resnet101_640/v19/'
FINE_TUNE_CHECKPOINT = './ObjectDetection/models/ssd_resnet101_640/v18/best_ckps/ckpt-178'

def _compute_losses_and_predictions_dicts(
    model, features, labels,
    add_regularization_loss=True):
  """Computes the losses dict and predictions dict for a model on inputs.

  Args:
    model: a DetectionModel (based on Keras).
    features: Dictionary of feature tensors from the input dataset.
      Should be in the format output by `inputs.train_input` and
      `inputs.eval_input`.
        features[fields.InputDataFields.image] is a [batch_size, H, W, C]
          float32 tensor with preprocessed images.
        features[HASH_KEY] is a [batch_size] int32 tensor representing unique
          identifiers for the images.
        features[fields.InputDataFields.true_image_shape] is a [batch_size, 3]
          int32 tensor representing the true image shapes, as preprocessed
          images could be padded.
        features[fields.InputDataFields.original_image] (optional) is a
          [batch_size, H, W, C] float32 tensor with original images.
    labels: A dictionary of groundtruth tensors post-unstacking. The original
      labels are of the form returned by `inputs.train_input` and
      `inputs.eval_input`. The shapes may have been modified by unstacking with
      `model_lib.unstack_batch`. However, the dictionary includes the following
      fields.
        labels[fields.InputDataFields.num_groundtruth_boxes] is a
          int32 tensor indicating the number of valid groundtruth boxes
          per image.
        labels[fields.InputDataFields.groundtruth_boxes] is a float32 tensor
          containing the corners of the groundtruth boxes.
        labels[fields.InputDataFields.groundtruth_classes] is a float32
          one-hot tensor of classes.
        labels[fields.InputDataFields.groundtruth_weights] is a float32 tensor
          containing groundtruth weights for the boxes.
        -- Optional --
        labels[fields.InputDataFields.groundtruth_instance_masks] is a
          float32 tensor containing only binary values, which represent
          instance masks for objects.
        labels[fields.InputDataFields.groundtruth_instance_mask_weights] is a
          float32 tensor containing weights for the instance masks.
        labels[fields.InputDataFields.groundtruth_keypoints] is a
          float32 tensor containing keypoints for each box.
        labels[fields.InputDataFields.groundtruth_dp_num_points] is an int32
          tensor with the number of sampled DensePose points per object.
        labels[fields.InputDataFields.groundtruth_dp_part_ids] is an int32
          tensor with the DensePose part ids (0-indexed) per object.
        labels[fields.InputDataFields.groundtruth_dp_surface_coords] is a
          float32 tensor with the DensePose surface coordinates.
        labels[fields.InputDataFields.groundtruth_group_of] is a tf.bool tensor
          containing group_of annotations.
        labels[fields.InputDataFields.groundtruth_labeled_classes] is a float32
          k-hot tensor of classes.
        labels[fields.InputDataFields.groundtruth_track_ids] is a int32
          tensor of track IDs.
        labels[fields.InputDataFields.groundtruth_keypoint_depths] is a
          float32 tensor containing keypoint depths information.
        labels[fields.InputDataFields.groundtruth_keypoint_depth_weights] is a
          float32 tensor containing the weights of the keypoint depth feature.
    add_regularization_loss: Whether or not to include the model's
      regularization loss in the losses dictionary.

  Returns:
    A tuple containing the losses dictionary (with the total loss under
    the key 'Loss/total_loss'), and the predictions dictionary produced by
    `model.predict`.

  """
  model_lib.provide_groundtruth(model, labels)
  preprocessed_images = features[fields.InputDataFields.image]

  prediction_dict = model.predict(
      preprocessed_images,
      features[fields.InputDataFields.true_image_shape],
      **model.get_side_inputs(features))
  prediction_dict = ops.bfloat16_to_float32_nested(prediction_dict)

  losses_dict = model.loss(
      prediction_dict, features[fields.InputDataFields.true_image_shape])
  losses = [loss_tensor for loss_tensor in losses_dict.values()]
  if add_regularization_loss:
    # TODO(kaftan): As we figure out mixed precision & bfloat 16, we may
    ## need to convert these regularization losses from bfloat16 to float32
    ## as well.
    regularization_losses = model.regularization_losses()
    if regularization_losses:
      regularization_losses = ops.bfloat16_to_float32_nested(
          regularization_losses)
      regularization_loss = tf.add_n(
          regularization_losses, name='regularization_loss')
      losses.append(regularization_loss)
      losses_dict['Loss/regularization_loss'] = regularization_loss

  total_loss = tf.add_n(losses, name='total_loss')
  losses_dict['Loss/total_loss'] = total_loss

  return losses_dict, prediction_dict

def validate_tf_v2_checkpoint_restore_map(checkpoint_restore_map):
  """Ensure that given dict is a valid TF v2 style restore map.

  Args:
    checkpoint_restore_map: A nested dict mapping strings to
      tf.keras.Model objects.

  Raises:
    ValueError: If they keys in checkpoint_restore_map are not strings or if
      the values are not keras Model objects.

  """

  for key, value in checkpoint_restore_map.items():
    if not (isinstance(key, str) and
            (isinstance(value, tf.Module)
             or isinstance(value, tf.train.Checkpoint))):
      if isinstance(key, str) and isinstance(value, dict):
        validate_tf_v2_checkpoint_restore_map(value)

def load_fine_tune_checkpoint(model, checkpoint_path, checkpoint_type,
                              checkpoint_version, run_model_on_dummy_input,
                              input_dataset, unpad_groundtruth_tensors):
  """Load a fine tuning classification or detection checkpoint.

  To make sure the model variables are all built, this method first executes
  the model by computing a dummy loss. (Models might not have built their
  variables before their first execution)

  It then loads an object-based classification or detection checkpoint.

  This method updates the model in-place and does not return a value.

  Args:
    model: A DetectionModel (based on Keras) to load a fine-tuning
      checkpoint for.
    checkpoint_path: Directory with checkpoints file or path to checkpoint.
    checkpoint_type: Whether to restore from a full detection
      checkpoint (with compatible variable names) or to restore from a
      classification checkpoint for initialization prior to training.
      Valid values: `detection`, `classification`.
    checkpoint_version: train_pb2.CheckpointVersion.V1 or V2 enum indicating
      whether to load checkpoints in V1 style or V2 style.  In this binary
      we only support V2 style (object-based) checkpoints.
    run_model_on_dummy_input: Whether to run the model on a dummy input in order
      to ensure that all model variables have been built successfully before
      loading the fine_tune_checkpoint.
    input_dataset: The tf.data Dataset the model is being trained on. Needed
      to get the shapes for the dummy loss computation.
    unpad_groundtruth_tensors: A parameter passed to unstack_batch.

  Raises:
    IOError: if `checkpoint_path` does not point at a valid object-based
      checkpoint
    ValueError: if `checkpoint_version` is not train_pb2.CheckpointVersion.V2
  """
  if checkpoint_version == train_pb2.CheckpointVersion.V1:
    raise ValueError('Checkpoint version should be V2')

  if run_model_on_dummy_input:
    _ensure_model_is_built(model, input_dataset, unpad_groundtruth_tensors)

  restore_from_objects_dict = model.restore_from_objects(
      fine_tune_checkpoint_type=checkpoint_type)
  validate_tf_v2_checkpoint_restore_map(restore_from_objects_dict)
  ckpt = tf.train.Checkpoint(**restore_from_objects_dict)
  ckpt.restore(
      checkpoint_path).expect_partial().assert_existing_objects_matched()

def get_filepath(strategy, filepath):
  """Get appropriate filepath for worker.

  Args:
    strategy: A tf.distribute.Strategy object.
    filepath: A path to where the Checkpoint object is stored.

  Returns:
    A temporary filepath for non-chief workers to use or the original filepath
    for the chief.
  """
  if strategy.extended.should_checkpoint:
    return filepath
  else:
    # TODO(vighneshb) Replace with the public API when TF exposes it.
    task_id = strategy.extended._task_id  # pylint:disable=protected-access
    return os.path.join(filepath, 'temp_worker_{:03d}'.format(task_id))

def _ensure_model_is_built(model, input_dataset, unpad_groundtruth_tensors):
  """Ensures that model variables are all built, by running on a dummy input.

  Args:
    model: A DetectionModel to be built.
    input_dataset: The tf.data Dataset the model is being trained on. Needed to
      get the shapes for the dummy loss computation.
    unpad_groundtruth_tensors: A parameter passed to unstack_batch.
  """
  features, labels = iter(input_dataset).next()

  @tf.function
  def _dummy_computation_fn(features, labels):
    model._is_training = False  # pylint: disable=protected-access
    tf.keras.backend.set_learning_phase(False)

    labels = model_lib.unstack_batch(
        labels, unpad_groundtruth_tensors=unpad_groundtruth_tensors)

    return _compute_losses_and_predictions_dicts(model, features, labels)

  strategy = tf.compat.v2.distribute.get_strategy()
  if hasattr(tf.distribute.Strategy, 'run'):
    strategy.run(
        _dummy_computation_fn, args=(
            features,
            labels,
        ))
  else:
    strategy.experimental_run_v2(
        _dummy_computation_fn, args=(
            features,
            labels,
        ))

def normalize_dict(values_dict, num_replicas):
  num_replicas = tf.constant(num_replicas, dtype=tf.float32)
  return {key: tf.math.divide(loss, num_replicas) for key, loss
          in values_dict.items()}

def eager_train_step(detection_model,
                     features,
                     labels,
                     unpad_groundtruth_tensors,
                     optimizer,
                     add_regularization_loss=True,
                     clip_gradients_value=None,
                     num_replicas=1.0):
  """Process a single training batch.

  This method computes the loss for the model on a single training batch,
  while tracking the gradients with a gradient tape. It then updates the
  model variables with the optimizer, clipping the gradients if
  clip_gradients_value is present.

  This method can run eagerly or inside a tf.function.

  Args:
    detection_model: A DetectionModel (based on Keras) to train.
    features: Dictionary of feature tensors from the input dataset.
      Should be in the format output by `inputs.train_input.
        features[fields.InputDataFields.image] is a [batch_size, H, W, C]
          float32 tensor with preprocessed images.
        features[HASH_KEY] is a [batch_size] int32 tensor representing unique
          identifiers for the images.
        features[fields.InputDataFields.true_image_shape] is a [batch_size, 3]
          int32 tensor representing the true image shapes, as preprocessed
          images could be padded.
        features[fields.InputDataFields.original_image] (optional, not used
          during training) is a
          [batch_size, H, W, C] float32 tensor with original images.
    labels: A dictionary of groundtruth tensors. This method unstacks
      these labels using model_lib.unstack_batch. The stacked labels are of
      the form returned by `inputs.train_input` and `inputs.eval_input`.
        labels[fields.InputDataFields.num_groundtruth_boxes] is a [batch_size]
          int32 tensor indicating the number of valid groundtruth boxes
          per image.
        labels[fields.InputDataFields.groundtruth_boxes] is a
          [batch_size, num_boxes, 4] float32 tensor containing the corners of
          the groundtruth boxes.
        labels[fields.InputDataFields.groundtruth_classes] is a
          [batch_size, num_boxes, num_classes] float32 one-hot tensor of
          classes. num_classes includes the background class.
        labels[fields.InputDataFields.groundtruth_weights] is a
          [batch_size, num_boxes] float32 tensor containing groundtruth weights
          for the boxes.
        -- Optional --
        labels[fields.InputDataFields.groundtruth_instance_masks] is a
          [batch_size, num_boxes, H, W] float32 tensor containing only binary
          values, which represent instance masks for objects.
        labels[fields.InputDataFields.groundtruth_instance_mask_weights] is a
          [batch_size, num_boxes] float32 tensor containing weights for the
          instance masks.
        labels[fields.InputDataFields.groundtruth_keypoints] is a
          [batch_size, num_boxes, num_keypoints, 2] float32 tensor containing
          keypoints for each box.
        labels[fields.InputDataFields.groundtruth_dp_num_points] is a
          [batch_size, num_boxes] int32 tensor with the number of DensePose
          sampled points per instance.
        labels[fields.InputDataFields.groundtruth_dp_part_ids] is a
          [batch_size, num_boxes, max_sampled_points] int32 tensor with the
          part ids (0-indexed) for each instance.
        labels[fields.InputDataFields.groundtruth_dp_surface_coords] is a
          [batch_size, num_boxes, max_sampled_points, 4] float32 tensor with the
          surface coordinates for each point. Each surface coordinate is of the
          form (y, x, v, u) where (y, x) are normalized image locations and
          (v, u) are part-relative normalized surface coordinates.
        labels[fields.InputDataFields.groundtruth_labeled_classes] is a float32
          k-hot tensor of classes.
        labels[fields.InputDataFields.groundtruth_track_ids] is a int32
          tensor of track IDs.
        labels[fields.InputDataFields.groundtruth_keypoint_depths] is a
          float32 tensor containing keypoint depths information.
        labels[fields.InputDataFields.groundtruth_keypoint_depth_weights] is a
          float32 tensor containing the weights of the keypoint depth feature.
    unpad_groundtruth_tensors: A parameter passed to unstack_batch.
    optimizer: The training optimizer that will update the variables.
    add_regularization_loss: Whether or not to include the model's
      regularization loss in the losses dictionary.
    clip_gradients_value: If this is present, clip the gradients global norm
      at this value using `tf.clip_by_global_norm`.
    num_replicas: The number of replicas in the current distribution strategy.
      This is used to scale the total loss so that training in a distribution
      strategy works correctly.

  Returns:
    The total loss observed at this training step
  """
  # """Execute a single training step in the TF v2 style loop."""
  is_training = True

  detection_model._is_training = is_training  # pylint: disable=protected-access
  tf.keras.backend.set_learning_phase(is_training)

  labels = model_lib.unstack_batch(
      labels, unpad_groundtruth_tensors=unpad_groundtruth_tensors)

  with tf.GradientTape() as tape:
    losses_dict, _ = _compute_losses_and_predictions_dicts(
        detection_model, features, labels, add_regularization_loss)

    losses_dict = normalize_dict(losses_dict, num_replicas)

  trainable_variables = detection_model.trainable_variables

  total_loss = losses_dict['Loss/total_loss']
  gradients = tape.gradient(total_loss, trainable_variables)

  if clip_gradients_value:
    gradients, _ = tf.clip_by_global_norm(gradients, clip_gradients_value)
  optimizer.apply_gradients(zip(gradients, trainable_variables))

  return losses_dict

def reduce_dict(strategy, reduction_dict, reduction_op):
  # TODO(anjalisridhar): explore if it is safe to remove the # num_replicas
  # scaling of the loss and switch this to a ReduceOp.Mean
  return {
      name: strategy.reduce(reduction_op, loss, axis=None)
      for name, loss in reduction_dict.items()
  }

pipeline_config_path=PIPELINE_CONFIG_PATH
model_dir=MODEL_DIR,
train_steps=None
use_tpu=False,
checkpoint_every_n=100,
record_summaries=True

## Parse the configs
get_configs_from_pipeline_file = MODEL_BUILD_UTIL_MAP[
    'get_configs_from_pipeline_file']
merge_external_params_with_configs = MODEL_BUILD_UTIL_MAP[
    'merge_external_params_with_configs']
create_pipeline_proto_from_configs = MODEL_BUILD_UTIL_MAP[
    'create_pipeline_proto_from_configs']
steps_per_sec_list = []

configs = get_configs_from_pipeline_file(pipeline_config_path, config_override='train_input_reader {label_map_path: "./ObjectDetection/data/label_map.pbtxt" tf_record_input_reader {input_path: "./ObjectDetection/data/train-*.tfrecord"}}\
    train_config {fine_tune_checkpoint: "' + FINE_TUNE_CHECKPOINT + '"}')

kwargs = {}
kwargs.update({
    'train_steps': train_steps,
    'use_bfloat16': configs['train_config'].use_bfloat16 and use_tpu
})
configs = merge_external_params_with_configs(
    configs, None, kwargs_dict=kwargs)
model_config = configs['model']
train_config = configs['train_config']
train_input_config = configs['train_input_config']

unpad_groundtruth_tensors = train_config.unpad_groundtruth_tensors
add_regularization_loss = train_config.add_regularization_loss
clip_gradients_value = None
if train_config.gradient_clipping_by_norm > 0:
  clip_gradients_value = train_config.gradient_clipping_by_norm

# update train_steps from config but only when non-zero value is provided
if train_steps is None and train_config.num_steps != 0:
  train_steps = train_config.num_steps

if kwargs['use_bfloat16']:
  tf.compat.v2.keras.mixed_precision.set_global_policy('mixed_bfloat16')

if train_config.load_all_detection_checkpoint_vars:
  raise ValueError('train_pb2.load_all_detection_checkpoint_vars '
                     'unsupported in TF2')

config_util.update_fine_tune_checkpoint_type(train_config)
fine_tune_checkpoint_type = train_config.fine_tune_checkpoint_type
fine_tune_checkpoint_version = train_config.fine_tune_checkpoint_version
  
# Build the model, optimizer, and training input
strategy = tf.compat.v2.distribute.get_strategy()
with strategy.scope():
  detection_model = MODEL_BUILD_UTIL_MAP['detection_model_fn_base'](
      model_config=model_config, is_training=True,
      add_summaries=record_summaries)

def train_dataset_fn(input_context):
  """Callable to create train input."""
  # Create the inputs.
  train_input = inputs.train_input(
      train_config=train_config,
      train_input_config=train_input_config,
      model_config=model_config,
      model=detection_model,
      input_context=input_context)
  train_input = train_input.repeat()
  return train_input

train_input = strategy.experimental_distribute_datasets_from_function(train_dataset_fn)

global_step = tf.Variable(
    0, trainable=False, dtype=tf.compat.v2.dtypes.int64, name='global_step',
    aggregation=tf.compat.v2.VariableAggregation.ONLY_FIRST_REPLICA)

optimizer, (learning_rate,) = optimizer_builder.build(train_config.optimizer, global_step=global_step)

if train_config.optimizer.use_moving_average:
    _ensure_model_is_built(detection_model, train_input,
                     unpad_groundtruth_tensors)
    optimizer.shadow_copy(detection_model)

if callable(learning_rate):
  learning_rate_fn = learning_rate
else:
  learning_rate_fn = lambda: learning_rate

with strategy.scope():
    # Load a fine-tuning checkpoint.
    if train_config.fine_tune_checkpoint:
      load_fine_tune_checkpoint(
          detection_model, train_config.fine_tune_checkpoint,
          fine_tune_checkpoint_type, fine_tune_checkpoint_version,
          train_config.run_fine_tune_checkpoint_dummy_computation,
          train_input, unpad_groundtruth_tensors)
    ckpt = tf.compat.v2.train.Checkpoint(
        step=global_step, model=detection_model, optimizer=optimizer)
    manager_dir = get_filepath(strategy, model_dir)[0]
    manager = tf.compat.v2.train.CheckpointManager(
        ckpt, manager_dir, max_to_keep=1)

    ckpt.restore(SRC_CKPT).assert_existing_objects_matched()

    def train_step_fn(features, labels):
      """Single train step."""
      losses_dict = eager_train_step(
          detection_model,
          features,
          labels,
          unpad_groundtruth_tensors,
          optimizer,
          add_regularization_loss=add_regularization_loss,
          clip_gradients_value=clip_gradients_value,
          num_replicas=strategy.num_replicas_in_sync)
      global_step.assign_add(1)
      return losses_dict

    @tf.function()
    def _sample_and_train(strategy, train_step_fn, data_iterator):
      features, labels = data_iterator.next()
      if hasattr(tf.distribute.Strategy, 'run'):
        per_replica_losses_dict = strategy.run(
            train_step_fn, args=(features, labels))
      else:
        per_replica_losses_dict = (
            strategy.experimental_run_v2(
                train_step_fn, args=(features, labels)))
      return reduce_dict(strategy, per_replica_losses_dict, tf.distribute.ReduceOp.SUM)

    # run single step to initialize optimizer values
    train_input_iter = iter(train_input)
    losses_dict = _sample_and_train(strategy, train_step_fn, train_input_iter)

    # set step to 0
    ckpt._self_unconditional_checkpoint_dependencies[2].ref.assign(0)
    # set save_counter to 0
    ckpt._self_unconditional_checkpoint_dependencies[3].ref.assign(0)

    # reset all optimizer varibales
    for slot in ckpt._self_unconditional_checkpoint_dependencies[1].ref._slots.values():
      for var in slot.values():
        var.assign(tf.zeros_like(var.read_value()))

    ckpt.save(DEST_CKPT)