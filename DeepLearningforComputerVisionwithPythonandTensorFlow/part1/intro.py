#from https://www.youtube.com/watch?v=IA3WxTTPXqQ

import tensorflow as tf
import numpy as np

print('--------0 D---------')

tensor_zero_d = tf.constant(4)
print(tensor_zero_d)

print('--------ONE D---------')

tensor_one_d = tf.constant([2, 0., -3, 8 ,90.], dtype=tf.float32)
print(tensor_one_d)

casted_tensor_one_d = tf.cast(tensor_one_d, dtype=tf.bool)
print(casted_tensor_one_d)

tensor_bool = tf.constant([True, True, False ])
print(tensor_bool)

tensor_string = tf.constant(['Hello Wordl', "hi"])
print(tensor_string)

print('------- NP TO TENSOR ----------')

np_array = np.array([1, 2, 4])
print(np_array)

converted_tensor = tf.convert_to_tensor(np_array)
print(converted_tensor)

print('--------- TWO D --------')

tensor_two_d = tf.constant([
    [1,2,3],
    [3,5,-1],
    [1,5,6],
    [2,3,8]

])
print(tensor_two_d)

print('--------- THREE D --------')

tensor_three_d = tf.constant([
    [[1,2,0],
        [3,5,-1]],

    [[10,2,0],
        [1,0,2]],

    [[5,8,0],
     [2,7,8]],

    [[2,1,9],
    [4,-3,32]]
])
print(tensor_three_d)

print('--------- FOUR D --------')

tensor_four_d = tf.constant([
    [[[1,2,0],
        [3,5,-1]],
    [[2,1,9],
    [4,-3,32]],
    [[2,1,9],
    [4,-3,32]],
    [[5,8,0],
     [2,7,8]]],

    [[[2,4,5],
     [-4,3,2]],
     [[4,5,6],
      [2,-1,3]],
        [[4,5,6],
      [2,-1,3]],
    [[5,8,0],
     [2,7,8]]],

    [[[10,2,0],
        [1,0,2]],
    [[5,8,0],
     [2,7,8]],
     [[5,8,0],
     [2,7,8]],
    [[5,8,0],
     [2,7,8]]]
])
print(tensor_four_d)

print("------------ EYE TENSOR ----------------")

eye_tensor = tf.eye(
    num_rows=5 ,
    num_columns=None,
    batch_shape=[2, 4],
    dtype=tf.dtypes.float32,
    name=None
)

print(eye_tensor)

print("------------ FIll method ----------------")

fill_tensor = tf.fill([1, 3, 4], 5, name=None)
print(fill_tensor)

print("------------ Ones method / ones like ----------------")

ones_tensor = tf.ones(
    [5, 3, 2],
    dtype=tf.dtypes.float32,
    name=None
)
print(ones_tensor)

ones_like_tensor = tf.ones_like(fill_tensor)
print(ones_like_tensor)

print("------------ Zeros method ----------------")

zeros_tensor = tf.zeros(
    [3, 2],
    dtype=tf.dtypes.float32,
    name=None
)
print(zeros_tensor)

print("------------ shape method ----------------")

print(tf.shape(tensor_three_d))

print("------------ rank ----------------")
# shape of tensor 't' is [2, 2, 3]
t = tf.constant([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]])
print(tf.rank(t))

print("------------ size method ----------------")
print(tf.size(t, out_type=tf.float32))

print("------------ SHAPES ----------------")

print(tensor_zero_d.shape)
print(tensor_one_d.shape)
print(tensor_two_d.shape)
print(tensor_three_d.shape)

print("------------- NDIM  ---------------")

print(tensor_three_d.ndim)

print("------------ tf random ----------------")
random_tensor = tf.random.normal(
    [3, 2],
    mean=0.0,
    stddev=1.0,
    dtype=tf.dtypes.float32,
    seed=None,
    name=None
)

print(random_tensor)

print("------------ tf random uniform ----------------")
random_tensor_uniform = tf.random.uniform(
    [5,5],
    minval=0,
    maxval=1000,
    dtype=tf.dtypes.int32,
    seed=None,
    name=None
)

print(random_tensor_uniform)

tf.random.set_seed(5)
print(tf.random.uniform(shape=[3,], maxval=5, dtype=tf.int32, seed=10))
print(tf.random.uniform(shape=[3,], maxval=5, dtype=tf.int32, seed=10))
print(tf.random.uniform(shape=[3,], maxval=5, dtype=tf.int32, seed=10))
print(tf.random.uniform(shape=[3,], maxval=5, dtype=tf.int32, seed=1))