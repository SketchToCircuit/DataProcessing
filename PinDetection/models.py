import tensorflow as tf
from tensorflow.keras import layers

import config

def getModel():
    input1 = tf.keras.Input(shape=(config.IMG_SIZE,config.IMG_SIZE,1), name='input1')
    input2 = tf.keras.Input(shape=(max([val[1] for val in config.LABEL_CONVERT_DICT.values()])), name='input2')

    y = layers.Conv2D(64, (3,3), activation='relu')(input1)
    y = layers.Dropout(0.6)(y)
    y = layers.Conv2D(32, (3,3), activation='relu')(y)
    y = layers.Dropout(0.6)(y)
    y = layers.MaxPool2D()(y)
    y = layers.Flatten()(y)
    y = layers.Dropout(0.6)(y)
    y = layers.Dense(256, activation='relu')(y)

    x = layers.Dense(128, activation='relu')(input2)
    x = layers.Dropout(0.6)(x)
    x = layers.Dense(64 , activation='relu')(x)
    x = layers.Dense(32 , activation='relu')(x)
    x = layers.Dropout(0.6)(x)

    z = layers.Concatenate()([x, y])
    z = layers.Dense(128, activation='relu')(z)
    z = layers.Dropout(0.6)(z)
    z = layers.Dense(1024, activation='relu')(z)
    z = layers.Reshape((32,32,1))(z)

    model = tf.keras.Model(inputs = [input1, input2], outputs = z, name='predict')
    return model