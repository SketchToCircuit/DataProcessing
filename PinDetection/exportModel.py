import tensorflow as tf
import config
import models
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

model_dir = os.path.join(os.path.dirname(__file__), "test/")

version = 1
export_path = os.path.join(model_dir, str(version))

model = models.getModel()
if(os.path.exists(os.path.join(config.TRAINMODELPATH, "checkpoint"))):
    model.load_weights(config.TRAINMODELPATH)
model.summary()

tf.keras.models.save_model(
    model,
    export_path,
    overwrite=True,
    include_optimizer=True,
    save_format=None,
    signatures=None,
    options=None
)