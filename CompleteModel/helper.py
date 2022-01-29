import tensorflow as tf

def rotate_vector(vec: tf.Tensor, angle: tf.Tensor):
    rotation_matrix = tf.stack([tf.cos(angle),
                              -tf.sin(angle),  
                               tf.sin(angle),
                               tf.cos(angle)])
    rotation_matrix = tf.reshape(rotation_matrix, (2,2))
    return tf.matmul(vec, rotation_matrix)

def vector_length(vec: tf.Tensor):
    with tf.name_scope('vector_length'):
        return tf.sqrt(tf.reduce_sum(tf.square(vec), axis=-1, keepdims=True))

def normalize_vector(vec: tf.Tensor):
    with tf.name_scope('normalize_vector'):
        return vec / vector_length(vec)

def k_out_of_n_combinations(k: tf.Tensor, n: tf.Tensor):
    with tf.name_scope('k_out_of_n_combinations'):
        if k > n:
            tf.print("ERROR: n must be at least as big as k!!")
            return tf.constant([], tf.int32)

        # num_comb = n over k
        num_comb = tf.cast(tf.round(tf.exp(tf.math.lgamma(tf.cast(n + 1, tf.float32)) - tf.math.lgamma(tf.cast(n - k + 1, tf.float32)) - tf.math.lgamma(tf.cast(k + 1, tf.float32)))), tf.int32)

        def next_combination(c):
            # increment last index number if possible
            if c[-1] < n-1:
                c = c + tf.squeeze(tf.one_hot([k-1], k, dtype=tf.int32))
            else:
                # get last index number that is less than its folowing index number - 1
                shifted = tf.roll(c, -1, 0)
                mask = c < shifted - 1
                mask = tf.concat([mask[:-1], [False]], 0)
                last_satisfying_idx = tf.reduce_max(tf.cast(tf.where(mask), tf.int32))

                # set all following numbers in ascending order beginning with this incremented number
                x = c[last_satisfying_idx] + 1
                new_slice = tf.range(x, x + (k - last_satisfying_idx))
                update_indices = tf.expand_dims(tf.range(last_satisfying_idx, k), -1)
                c = tf.tensor_scatter_nd_update(c, update_indices, new_slice)
            return tf.reshape(c, [k])

        combinations = tf.TensorArray(tf.int32, size=num_comb)

        combination = tf.range(k)
        for i in tf.range(num_comb):
            combinations = combinations.write(i, combination)
            if i < num_comb - 1:
                combination = next_combination(combination)

        return combinations.stack()