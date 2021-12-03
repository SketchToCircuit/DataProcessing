
CUDA_VISIBLE_DEVICES=""
import numpy as np
import os

import tensorflow as tf
import tensorflow.keras.layers as layers

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

physical_devices = tf.config.list_physical_devices("GPU")
if len(physical_devices) <= 0:
    print("No GPU detected!")

#ToDo:
# -> DataSet pipeline
#   -> Import Arrays into DataSet
#       -> Load all pictures on runtime ?without augmentation?
#           -> Read images
#           -> Read position data from json
#   -> Data augmentation
#       -> Resize image with padding ?and save scale?
#       -> Rotate image
#           -> same
#       -> ?random noise over picture?
#       -> Noise on hint input
#       -> False positives on hint input
# -> Change to predict connections
#   -> Change NN to 2 inputs and 2 outputs
#   -> Change loss function for the second output (y+x^2)
#   ->  ?

BATCH_SIZE = 60
IMG_SIZE = 128
EPOCHS = 10

#def main():
#     dataset = createDS.GetDataSet()
#     model = getModel()
#     for i in dataset.take(1):
#         print(i)
#    keras.utils.plot_model(
#         model
#     )
#     model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
#     model.fit(dataset, epochs=EPOCHS)


def getModel():
    input1 = tf.keras.Input(shape=(128,128,1), name='input1')
    input2 = tf.keras.Input(shape=(16), name='input2')

    y = layers.Conv2D(64, 3, activation='relu')(input1)
    y = layers.Conv2D(32, 3, activation='relu')(y)
    y = layers.MaxPool2D()(y)
    y = layers.Flatten()(y)
    y = layers.Dense(128, activation='relu')(y)

    x = layers.Dense(128, activation='relu')(input2)
    x = layers.Dense(64 , activation='relu')(x)
    x = layers.Dense(32 , activation='relu')(x)

    z = layers.Concatenate()([x, y])
    z = layers.Dense(128, activation='relu')(z)
    h = layers.Dense(128, activation='relu')(z)

    z = layers.Dense(46, activation='softmax', name='output1')(z)
    h = layers.Dense(6, activation='relu', name='output2')(h)

    model = tf.keras.Model(inputs = [input1, input2], outputs = [z,h], name='predict')
    return model