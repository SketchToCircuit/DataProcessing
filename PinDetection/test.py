from xml.etree.ElementTree import PI
import models
import config
import tensorflow as tf
import os

# summary_writer = tf.summary.create_file_writer(os.path.join(config.LOGDIR,"test"))

model = models.getModel()
model.load_weights(config.TRAINMODELPATH)

tf.keras.utils.plot_model(model, 
                          to_file="./PinDetection/test.png", 
                          show_shapes=True,
                          show_dtype=False,
                          show_layer_names=True,
                          expand_nested=False,
                          dpi=96,
                          layer_range=None)

# image = tf.io.read_file("/mnt/hdd2/Sketch2Circuit/DataProcessing/PinDetection/test1.png")
# image = tf.image.decode_png(image)
# image = tf.image.rgb_to_grayscale(image)
# image = tf.math.divide(image, 255)

# label = tf.zeros(max([val[1] for val in config.LABEL_CONVERT_DICT.values()]), dtype=tf.int32)

# image = tf.expand_dims(image, 0)
# label= tf.expand_dims(label, 0)

# image.set_shape([1,config.IMG_SIZE,config.IMG_SIZE,1])
# label.set_shape([1,max([val[1] for val in config.LABEL_CONVERT_DICT.values()])])

# res = model.predict({"input1": image, "input2": label})

# with summary_writer.as_default():
#     tf.summary.image("test", res, step=0)