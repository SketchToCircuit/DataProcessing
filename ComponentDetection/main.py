import cv2
import tensorflow as tf
import numpy as np
import os
import json
import config

np.set_printoptions(precision=4)

def main():
    dataSet = tf.data.Dataset.from_generator(jsonGenerator, output_types=(tf.string, tf.int32, tf.int32, tf.double))
    dataSet = dataSet.map(loadImage).map(dataProc)
    # for img, hint, label, pins in dataSet.take(20):
    #     print(img)
    #     img = img.numpy()*255
    #     cv2.imwrite("test.png", img)

def dataProc(img, pins, label, hint):
    img = tf.cast(tf.bitwise.invert(img), dtype=tf.int32)
    img = tf.cast(tf.image.resize_with_pad(img,config.IMG_SIZE, config.IMG_SIZE), dtype=tf.int32)
    white = tf.ones((config.IMG_SIZE, config.IMG_SIZE, 1), dtype=tf.int32)*255
    img = tf.subtract(white, img)
    img = img/255

    return img, pins, label, hint

def loadImage(filepath, hint, label, pins):
    img = tf.io.read_file(filepath)
    img = tf.image.decode_png(img)

    label = tf.one_hot(label, len(config.CATEGORIES), dtype=tf.int32)
    hint = tf.one_hot(hint, len(config.HINTS), dtype=tf.int32)

    return img, pins, label, hint

def jsonGenerator():
    data = json.load(open(config.DATAJSONPATH))
    for component in data:
        for entry in data[component]:
            cmpPath = os.path.realpath(os.path.join(os.path.dirname(__file__), '..',entry["component_path"]))
            label = config.CATEGORIES.index(entry["type"])
            hint = config.HINTS.index(config.COMPONENTS[entry["type"]])
            pins = [[-1,-1],[-1,-1],[-1,-1]]
            count = 0
            for pinNmbr in entry["pins"]:
                pins[count][0] = entry["pins"][pinNmbr]["position"][0]
                pins[count][1] = entry["pins"][pinNmbr]["position"][1]
                count = count + 1
            pins = pins[0][0],pins[0][1],pins[1][0],pins[1][1],pins[2][0],pins[2][1]
            yield cmpPath, hint, label, pins

if __name__ == '__main__':
    main()