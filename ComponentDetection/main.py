from operator import le
import tensorflow as tf
import numpy as np
import os
from tensorflow import keras
from tensorflow._api.v2 import data
from tensorflow.keras import layers
from tensorflow.python.keras.engine.input_layer import Input
from tensorflow.python.keras.layers.advanced_activations import Softmax
from tensorflow.python.keras.layers.core import Dense
from tensorflow.python.ops.gen_array_ops import concat, shape
from tensorflow.python.ops.gen_batch_ops import batch

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

NUM_CATEGORIES = 46
NUM_HINTS = 15
IMG_SIZE = 128

EPOCHS = 100

dataset_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), "DataSet"))

def main():
    X, X_VAL , Y , Y_VAL , X_HINT , X_HINT_VAL = dataProc()
    model = getModel()
    keras.utils.plot_model(
        model
    )
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit([X_HINT, X], Y, epochs=EPOCHS, batch_size = BATCH_SIZE, validation_data=([X_HINT_VAL, X_VAL], Y_VAL))

def dataProc():
    X = np.load(os.path.join(dataset_dir, "X.npy"))
    X_HINT = np.load(os.path.join(dataset_dir, "X_HINT.npy"))
    Y = np.load(os.path.join(dataset_dir, "Y.npy"))

    X, X_HINT, Y = dataAugm(X, X_HINT, Y)

    X_HINT, X_HINT_VAL = np.split(X_HINT, [int(0.7*len(X_HINT))])
    X, X_VAL = np.split(X, [int(0.7*len(X))])
    Y, Y_VAL =np.split(Y, [int(0.7*len(Y))])
    return X, X_VAL , Y , Y_VAL , X_HINT , X_HINT_VAL

def dataAugm(X, X_HINT, Y):
    return X, X_HINT, Y

def getModel():
    input0 = keras.Input(shape=(NUM_HINTS), name='hint')
    input1 = keras.Input(shape=(128,128,1), name='img')

    y = layers.Conv2D(64, 3, activation='relu')(input1)
    y = layers.Conv2D(32, 3, activation='relu')(y)
    y = layers.MaxPool2D()(y)
    y = layers.Flatten()(y)
    y = layers.Dense(128, activation='relu')(y)

    x = layers.Dense(128, activation='relu')(input0)
    x = layers.Dense(64 , activation='relu')(x)
    x = layers.Dense(32 , activation='relu')(x)

    z = layers.Concatenate()([x, y])
    z = layers.Dense(128, activation='relu')(z)
    z = layers.Dense(NUM_CATEGORIES, activation='softmax', name='output')(z)

    model = keras.Model(inputs = [input0, input1], outputs = z, name="predict")
    return model

if __name__ == '__main__':
    main()