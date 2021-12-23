import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import tensorflow_addons as tfa
import cv2
import numpy as np

def resize_antialiased(img, size, method, antialias):
    if antialias:
        return tf.image.resize(img, size, method, antialias=True)
    else:
        return tf.image.resize(img, size, method, antialias=False)

def threshold(img):
    return tf.where(img < 200, 0.0, 255.0)

def contrast_boost(img):
    return tf.clip_by_value(tf.image.adjust_contrast(img, tf.random.uniform(shape=[], minval=1.5, maxval=3.0)), 0.0, 255.0)

def dilate(img, size):
    # https://stackoverflow.com/questions/54686895/tensorflow-dilation-behave-differently-than-morphological-dilation
    # image needs another dimension for a batchszie of 1
    img = tf.expand_dims(img, axis=0)

    # max kernel_size = 10
    # create kernel with random size with shape (a, a, 3)
    size = tf.clip_by_value(size, 1, 10)
    size = tf.repeat(size, 2)
    size = tf.pad(size, paddings=tf.constant([[0, 1]]), constant_values=3)
    kernel = tf.ensure_shape(tf.zeros(size, dtype=tf.float32), [None, None, 3])

    img = tf.nn.dilation2d(img, filters=kernel, strides=(1,1,1,1), dilations=(1,1,1,1), padding="SAME", data_format="NHWC")

    # revert batch dimension
    img = tf.squeeze(img, axis=0)

    return img

def erode(img, size):
    # https://stackoverflow.com/questions/54686895/tensorflow-dilation-behave-differently-than-morphological-dilation
    # image needs another dimension for a batchszie of 1
    img = tf.expand_dims(img, axis=0)

    # max kernel_size = 10
    # create kernel with random size with shape (a, a, 3)
    size = tf.clip_by_value(size, 1, 10)
    size = tf.repeat(size, 2)
    size = tf.pad(size, paddings=tf.constant([[0, 1]]), constant_values=3)
    kernel = tf.ensure_shape(tf.zeros(size, dtype=tf.float32), [None, None, 3])

    img = tf.nn.erosion2d(img, filters=kernel, strides=(1,1,1,1), dilations=(1,1,1,1), padding="SAME", data_format="NHWC")

    # revert batch dimension
    img = tf.squeeze(img, axis=0)

    return img

def warp_random(img, strength):
    # https://www.tensorflow.org/addons/tutorials/image_ops#dense_image_warp
    img = tf.expand_dims(img, 0)
    img_dims = tf.slice(tf.shape(img), [0], [3])
    flow_shape = tf.concat([img_dims, tf.constant([2])], 0)
    # flow shape is no a tensor with the values [1, height, width, 2]
    rand_flow = tf.random.normal(flow_shape, stddev=strength) # standard devaition = strength of effect
    img = tfa.image.dense_image_warp(img, rand_flow)
    return tf.squeeze(img, 0)

def warp_sinusoidal(img, strength):
    #TODO
    return img

def noise_normal(img, strength):
    img = img + tf.random.normal(tf.shape(img), mean=0.0, stddev=strength, dtype=tf.float32)
    img = tf.clip_by_value(img, 0, 255)
    return img

def noise_uniform(img, strength):
    img = img + tf.random.uniform(tf.shape(img), -strength, strength, dtype=tf.dtypes.float32)
    img = tf.clip_by_value(img, 0, 255)
    return img

def uneven_resize(img, span):
    # 50% probability for scaling height 50/50
    if tf.random.uniform([]) < 0.5:
        newsize = [
        tf.cast(tf.cast(tf.shape(img)[1],tf.dtypes.float32) * tf.random.uniform([],minval=1 -span, maxval=1 + span),tf.dtypes.int32),
        tf.shape(img)[0]]
    else:
         newsize = [
        tf.cast(tf.cast(tf.shape(img)[0],tf.dtypes.float32) * tf.random.uniform([],minval=1 -span, maxval=1 + span),tf.dtypes.int32),
        tf.shape(img)[1]]
    
    r = tf.random.uniform([])
    if r < 1/3:
        img = resize_antialiased(img, newsize, tf.image.ResizeMethod.AREA, tf.random.uniform([]) < 0.5)
    elif r < 2/3:
        img = resize_antialiased(img, newsize, tf.image.ResizeMethod.BICUBIC, tf.random.uniform([]) < 0.5)
    else:
        img = resize_antialiased(img, newsize, tf.image.ResizeMethod.NEAREST_NEIGHBOR, tf.random.uniform([]) < 0.5)

    return img  

def shearing(img, boxes, shearlevel):
    img = tfa.image.shear_x(img,level=shearlevel, replace=[255, 255, 255])
    #new bounding boxes are the sheared diagonal corner points of the bounding box
    #Points are counteed counter cockwise beginning with left Top
    boundingPoints = [tf.constant(boxes[0], boxes[1]), tf.constant(boxes[2], boxes[1]), tf.constant(boxes[2], boxes[3]), tf.constant(boxes[0], boxes[3])]
    for i in range(4):
        boundingPoints[i] = tf.matmul(boundingPoints[i], tf.constant([shearlevel,1],[0,1]))

    boxes[0] = tf.minimum(boundingPoints[0],)
    return img, boxes 

@tf.function
def augment(image, boxes):
    '''
    image: Tensor("", shape=(None, None, 3), dtype=float32) with values in [0, 255]
    boxes: Tensor("", shape=(None, 4), dtype=float32) every item is in form of [ymin, xmin, ymax, xmax] where the coordinates are in [0, 1] (normalized to image size)
    '''
    image = tf.ensure_shape(image, [None, None, 3])

    # 70% contrast boosting or 30% threshold
    if tf.random.uniform([]) < 0.7:
        image = contrast_boost(image)
    else:
        image = threshold(image)

    # 70% dilation or erosion
    if tf.random.uniform([]) < 0.7:
        # 90% erosion, 10% dilation
        if tf.random.uniform([]) < 0.9:
            image = erode(image, tf.random.uniform(shape=[], minval=1, maxval=6, dtype=tf.int64)) # between 1 and 5 (inclusive) for erosion (thicker)
        else:
            image = dilate(image, 2) # kernel size for dilation (smaller) is always 2

    # 30% image warping
    if tf.random.uniform([]) < 0.3:
        # 50% random (normal distributed) warping
        if tf.random.uniform([]) < 0.5:
            image = warp_random(image, tf.random.uniform([], minval=0.2, maxval=1.5)) # random strength
        # 50% sinusoidal warping (not implemented yet)
        else:
            image = warp_sinusoidal(image, tf.random.uniform([], minval=0, maxval=1)) # random strength

    # 40% shearing
    if tf.random.uniform([]) < 1:
        image, boxes = shearing(image, boxes, 0.4)

    # # 70% add  Noise
    if tf.random.uniform([]) < 7.0:
        # 70% add normal Noise
        if tf.random.uniform([]) < 0.7:
            image = noise_normal(image, strength=tf.random.uniform([], minval=10, maxval=30))
        # 30% add uniform noise
        else:
            image = noise_uniform(image, strength=tf.random.uniform([], minval=10, maxval=30))

    # # 30% resize Picture uneven
    if tf.random.uniform([]) < 0.3:
        image = uneven_resize(image, span=0.1)         

    #set all color chanels to the same value 
    image = tf.repeat(tf.image.rgb_to_grayscale(image), 3, axis=-1)
    return image, boxes

# for eagerly testing the augmentation on *.tfrecord
def test(path: str, num_samples: int):
    dataset = tf.data.TFRecordDataset(path)

    ft_desc = {
        'image/height': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'image/width': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'image/filename': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'image/source_id': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'image/encoded': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'image/format': tf.io.FixedLenFeature([], tf.string, default_value=b'jpeg'),
        'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
        'image/object/class/text': tf.io.VarLenFeature(tf.string),
        'image/object/class/label': tf.io.VarLenFeature(tf.int64)}

    dataset = dataset.map(lambda example: tf.io.parse_single_example(example, ft_desc))

    def augment_dataset(example):
        boxes = tf.stack([example['image/object/bbox/ymin'].values, example['image/object/bbox/xmin'].values, example['image/object/bbox/ymax'].values, example['image/object/bbox/xmax'].values], axis=1)
        img = tf.cast(tf.io.decode_jpeg(example['image/encoded']), dtype=tf.float32)

        img, boxes = augment(img, boxes)

        return img, boxes

    dataset = dataset.map(augment_dataset)

    for img, boxes in dataset.take(num_samples):
        img = img.numpy().astype(np.uint8)

        for box in boxes.numpy():
            xmin = box[1] * img.shape[1]
            ymin = box[0] * img.shape[0]
            xmax = box[3] * img.shape[1]
            ymax = box[2] * img.shape[0]
        
            xmin = int(xmin)
            ymin = int(ymin)
            xmax = int(xmax)
            ymax = int(ymax)

            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), thickness=3)

        cv2.imshow('', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    test('./ObjectDetection/data/train-0.tfrecord', 5)