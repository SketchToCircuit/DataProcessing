import tensorflow as tf
with tf.Graph().as_default() as g:
    input = tf.compat.v1.placeholder(tf.string, shape=[])