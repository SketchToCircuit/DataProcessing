import os

# magic optimization
os.environ['CUDA_CACHE_DISABLE'] = '0'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

import tensorflow as tf

physical_devices = tf.config.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("No GPU detected!")

tf.config.optimizer.set_experimental_options({
    'layout_optimizer': True,
    'constant_folding': True,
    'shape_optimization': True,
    'remapping': True,
    'arithmetic_optimization': True,
    'dependency_optimization': True,
    'loop_optimization': True,
    'function_optimization': True,
    'debug_stripper': True,
    'scoped_allocator_optimization': True,
    'implementation_selector': True,
    'disable_meta_optimizer': False
})

from combined_model import CombinedModel

model = CombinedModel('./ObjectDetection/exported_models/ssd_resnet101_640_v14/saved_model', './PinDetection/exported/1')

signature = {tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY : model.__call__.get_concrete_function(tf.TensorSpec((None), dtype=tf.string))}
tf.saved_model.save(model, './CompleteModel/Exported/1', signature)