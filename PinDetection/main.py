import tensorflow as tf
from tensorboard import program
import os
import json
from tensorflow import keras
from tensorflow.keras import layers
from random import randint, randrange
import tensorflow_addons as tfa

import config
import models

physical_devices = tf.config.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("No GPU detected!")

def main():
    tbDataSet = tf.data.Dataset.from_generator(tbjsonGenerator, output_types=(tf.string, tf.int32, tf.double))
    tbDataSet = tbDataSet.map(loadImage,num_parallel_calls=tf.data.AUTOTUNE).map(dataProc,num_parallel_calls=tf.data.AUTOTUNE)
    tbDataSet = tbDataSet.map(dataAugment,num_parallel_calls=tf.data.AUTOTUNE).batch(len(list(tbDataSet)), num_parallel_calls=tf.data.AUTOTUNE, deterministic=False).prefetch(10)

    dataSet = tf.data.Dataset.from_generator(jsonGenerator, output_types=(tf.string, tf.int32, tf.double))
    dataSet = dataSet.map(loadImage,num_parallel_calls=tf.data.AUTOTUNE).map(dataProc,num_parallel_calls=tf.data.AUTOTUNE)
    train_size = int(0.7*len(list(dataSet)))
    dataSet = dataSet.shuffle(train_size+200,seed=12)
    trainDs = dataSet.take(train_size).map(dataAugment,num_parallel_calls=tf.data.AUTOTUNE)
    trainDs = trainDs.batch(128, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False).prefetch(512)
    valDs = dataSet.skip(train_size).map(noAugment,num_parallel_calls=tf.data.AUTOTUNE)
    valDs = valDs.batch(128, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False).prefetch(512)

    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=config.LOGDIR,update_freq=1,write_graph=True, write_images=True,histogram_freq=1)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=config.TRAINMODELPATH,save_freq=len(list(trainDs))*10,save_weights_only=True,verbose=1)

    model = models.getModel()
    if(os.path.exists(os.path.join(config.TRAINMODELPATH, "checkpoint"))):
        model.load_weights(config.TRAINMODELPATH)
    model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])
    model.summary()
    model.fit(trainDs, epochs=400, validation_data=valDs,callbacks=[tb_callback, cp_callback,CustomTensorboard(tbDataSet)])

class CustomTensorboard(keras.callbacks.Callback):
        def __init__(self,tbDataSet,patience=0):
            super(CustomTensorboard, self).__init__()
            self.patience = patience
            self.tbDataSet = tbDataSet
            self.summary_writer = tf.summary.create_file_writer(os.path.join(config.LOGDIR,"images"))
            self.sortedComps = getComponentsSorted()

        def on_epoch_end(self, epoch, logs=None):
            if(not (epoch % 5 == 0)): return
            inputs, outputs = zip(*self.tbDataSet)
            inputs = inputs[0]
            outputs = outputs[0]
            predic = tf.image.resize_with_pad(tf.cast(tf.reshape(self.model.predict({"input1": inputs["input1"], "input2": inputs["input2"]}), [-1,32,32,1]),tf.float32), 128,128, antialias=False)
            val = tf.image.resize_with_pad(tf.cast(outputs,tf.float32),128,128,antialias=False)
            component = tf.cast(inputs["input1"],tf.float32)
            count = 0
            with self.summary_writer.as_default():
                for _component, _val, _pinImg in zip(component, val, predic):
                    tf.summary.image(self.sortedComps[count], [_component, _val, _pinImg] , step=epoch)
                    count = count + 1

@tf.function
def noAugment(img, label, pinImage):
    return ({"input1": img, "input2": label}, pinImage)

@tf.function
def dataAugment(img, label, pinImage):
    #Turn Images randomly
    angle = tf.random.uniform((1,), -20,20, dtype=tf.float32)
    img = tfa.image.rotate(img, angle, fill_mode="nearest")
    pinImage = tfa.image.rotate(pinImage, angle, fill_mode="nearest")

    pinImage = tfa.image.gaussian_filter2d(pinImage, 24, 2.7)
    
    if(randint(1,4) == 1):
        pinImage = tf.image.flip_left_right(pinImage)
        img = tf.image.flip_left_right(img)
    
    if(randint(1,4) == 2):
        pinImage = tf.image.flip_up_down(pinImage)
        img = tf.image.flip_up_down(img)

    return ({"input1": img, "input2": label}, pinImage)

@tf.function
def dataProc(img, pins, label):
    oldWidth = tf.cast(tf.shape(img)[1], tf.float32)
    oldHeigt = tf.cast(tf.shape(img)[0], tf.float32)

    img = tf.cast(tf.bitwise.invert(img), dtype=tf.int32)
    img = tf.cast(tf.image.resize_with_pad(img,config.IMG_SIZE, config.IMG_SIZE), dtype=tf.int32)
    white = tf.ones((config.IMG_SIZE, config.IMG_SIZE, 1), dtype=tf.int32)*255
    img = tf.subtract(white, img)
    img = tf.divide(img, 255)

    pins = tf.cast(pins, tf.float32)
    box1 = tf.concat([tf.divide(pins[0][1],oldHeigt),tf.divide(pins[0][0],oldWidth),tf.divide(pins[0][1],oldHeigt),tf.divide(pins[0][0],oldWidth)], 0)
    box2 = tf.concat([tf.divide(pins[1][1],oldHeigt),tf.divide(pins[1][0],oldWidth),tf.divide(pins[1][1],oldHeigt),tf.divide(pins[1][0],oldWidth)], 0)
    box3 = tf.concat([tf.divide(pins[2][1],oldHeigt),tf.divide(pins[2][0],oldWidth),tf.divide(pins[2][1],oldHeigt),tf.divide(pins[2][0],oldWidth)], 0)


    box1 = tf.reshape(box1,[1,1,4])
    box2 = tf.reshape(box2,[1,1,4])
    box3 = tf.reshape(box3,[1,1,4])

    colors = tf.constant([[1, 1, 1], [1, 1, 1]], dtype=tf.float32)
    pinImage = tf.zeros([1,oldHeigt,oldWidth,1],dtype=tf.float32)

    if(pins[0][0] > 0) : pinImage = tf.image.draw_bounding_boxes(pinImage, box1, colors)
    if(pins[1][0] > 0) : pinImage = tf.image.draw_bounding_boxes(pinImage, box2, colors)
    if(pins[2][0] > 0) : pinImage = tf.image.draw_bounding_boxes(pinImage, box3, colors)

    pinImage = tf.image.resize_with_pad(pinImage, 32, 32, antialias=True)
    pinImage = tf.squeeze(pinImage)
    pinImage = tf.reshape(pinImage, [32,32,1])
    pinImage = tf.math.ceil(pinImage)

    pinImage = tfa.image.gaussian_filter2d(pinImage, 8, 3)
    pinImage = tf.math.ceil(pinImage)
    
    img.set_shape([128,128,1])
    label.set_shape([len(config.CATEGORIES),])
    pinImage.set_shape([32,32,1])

    return img, label, pinImage

@tf.function
def loadImage(filepath, label, pins):
    img = tf.io.read_file(filepath)
    img = tf.image.decode_png(img)

    label = tf.one_hot(label, len(config.CATEGORIES), dtype=tf.int32)

    return img, pins, label

def jsonGenerator():
    data = json.load(open(config.DATAJSONPATH))
    for component in data:
        for entry in data[component]:
            cmpPath = os.path.join("/mnt/hdd2/Sketch2Circuit/",os.path.relpath(entry["component_path"]))
            label = config.CATEGORIES.index(entry["type"])
            pins = [[-1,-1],[-1,-1],[-1,-1]]
            count = 0
            for pinNmbr in entry["pins"]:
                pins[count][0] = entry["pins"][pinNmbr]["position"][0]
                pins[count][1] = entry["pins"][pinNmbr]["position"][1]
                count = count + 1
            pins = [pins[0][0],pins[0][1]],[pins[1][0],pins[1][1]],[pins[2][0],pins[2][1]]
            yield cmpPath, label, pins

def tbjsonGenerator():
    data = json.load(open(config.DATAJSONPATH))
    for component in data:
        pins = None
        label = None
        cmpPath = None
        for entry in data[component]:
            cmpPath = os.path.join("/mnt/hdd2/Sketch2Circuit/",os.path.relpath(entry["component_path"]))
            label = config.CATEGORIES.index(entry["type"])
            pins = [[-1,-1],[-1,-1],[-1,-1]]
            count = 0
            for pinNmbr in entry["pins"]:
                pins[count][0] = entry["pins"][pinNmbr]["position"][0]
                pins[count][1] = entry["pins"][pinNmbr]["position"][1]
                count = count + 1
            pins = [pins[0][0],pins[0][1]],[pins[1][0],pins[1][1]],[pins[2][0],pins[2][1]]
            break
        yield cmpPath, label, pins

#Return 
def getComponentsSorted():
        data = json.load(open(config.DATAJSONPATH))
        comps = data.keys()
        return list(comps)

if __name__ == '__main__':
    os.mkdir(config.LOGDIR)
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', config.TBDIR])
    url = tb.launch()
    print(f"Tensorboard listening on {url}")
    main()