import tensorflow as tf
from tensorboard import program
import os
import json
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa

import config

physical_devices = tf.config.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("No GPU detected!")

#tf.debugging.experimental.enable_dump_debug_info(config.LOGDIR, tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)

def main():
    dataSet = tf.data.Dataset.from_generator(jsonGenerator, output_types=(tf.string, tf.int32, tf.double))
    dataSet = dataSet.map(loadImage).map(dataProc).shuffle(1000, seed=12)

    train_size = int(0.7*len(list(dataSet)))
    trainDs = dataSet.take(train_size).batch(128, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
    testDs = dataSet.skip(train_size).batch(1, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
    valDs = dataSet.skip(train_size).batch(128, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)

    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=config.LOGDIR,update_freq=1,write_graph=True, write_images=True,histogram_freq=1)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=config.MODELPATH,save_freq=len(list(trainDs))*10,save_weights_only=True,verbose=1)

    model = getModel()
    if(os.path.exists(config.MODELPATH)):
        model.load_weights(config.MODELPATH)
    model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])
    model.summary()
    model.fit(trainDs, epochs=100, validation_data=valDs,callbacks=[tb_callback, cp_callback,CustomTensorboard(testDs)])

class CustomTensorboard(keras.callbacks.Callback):
        def __init__(self,testDs,patience=0):
            super(CustomTensorboard, self).__init__()
            self.patience = patience
            self.testDs = testDs
            self.summary_writer = tf.summary.create_file_writer(os.path.join(config.LOGDIR,"images"))

        def on_epoch_end(self, epoch, logs=None):
            for input, output in self.testDs.take(1):
                predic =  tf.image.resize_with_pad(tf.cast(tf.reshape(tf.squeeze(self.model.predict(input)),[32,32,1]),tf.float32),128,128,antialias=False)
                val = tf.image.resize_with_pad(tf.cast(tf.squeeze(output, axis=0),tf.float32),128,128,antialias=False)
                component = tf.cast(tf.squeeze(input["input1"], axis=0),tf.float32)
                with self.summary_writer.as_default():
                    tf.summary.image('component', [component,val,predic], step=epoch)
def getModel():
    input1 = tf.keras.Input(shape=(128,128,1), name='input1')
    input2 = tf.keras.Input(shape=(46), name='input2')

    y = layers.Conv2D(64, (3,3), activation='relu')(input1)
    y = layers.Dropout(0.2)(y)
    y = layers.Conv2D(32, (3,3), activation='relu')(y)
    y = layers.Dropout(0.2)(y)
    y = layers.MaxPool2D()(y)
    y = layers.Flatten()(y)
    y = layers.Dropout(0.3)(y)
    y = layers.Dense(256, activation='relu')(y)

    x = layers.Dense(128, activation='relu')(input2)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64 , activation='relu')(x)
    x = layers.Dense(32 , activation='relu')(x)
    x = layers.Dropout(0.2)(x)

    z = layers.Concatenate()([x, y])
    z = layers.Dense(128, activation='relu')(z)
    z = layers.Dropout(0.2)(z)
    z = layers.Dense(1024, activation='relu')(z)
    z = layers.Reshape((32,32,1))(z)

    model = tf.keras.Model(inputs = [input1, input2], outputs = z, name='predict')
    return model

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
    pinImage = tfa.image.gaussian_filter2d(pinImage, 24, 3)
    
    img.set_shape([128,128,1])
    label.set_shape([len(config.CATEGORIES),])
    pinImage.set_shape([32,32,1])

    return ({"input1": img, "input2": label}, pinImage)

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

if __name__ == '__main__':
    os.mkdir(config.LOGDIR)
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', config.LOGDIR])
    url = tb.launch()
    print(f"Tensorboard listening on {url}")
    input()
    # main()