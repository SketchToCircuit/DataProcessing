import tensorflow as tf

tf.config.optimizer.set_jit(True)

physical_devices = tf.config.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("No GPU detected!")

from combined_model import CombinedModel

model = CombinedModel('./ObjectDetection/exported_models/ssd_resnet101_640_v14/saved_model', './PinDetection/exported/1')

signature = {tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY : model.__call__.get_concrete_function(tf.TensorSpec((None), dtype=tf.string))}
tf.saved_model.save(model, './CompleteModel/Exported/1', signature)