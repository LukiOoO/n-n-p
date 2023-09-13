import tensorflow as tf
import numpy as np


tensor_indexed = tf.constant([3,6,2,4,6,66,7])
print(tensor_indexed)
print(tensor_indexed[0:4])
print(tensor_indexed[1:6])
print(tensor_indexed[3:-3])
print(tensor_indexed[1:6:2])

print('tensor 2d indexed')
tensor_two_d = tf.constant([
    [1,2,0],
    [3,5,-1],
    [1,5,6],
    [2,3,8]

])
print(tensor_two_d[0:3,0:2])
print(tensor_two_d[2,:])
print(tensor_two_d[2,0])
print(tensor_two_d[2,1:])
print(tensor_two_d[1:3, 0])
# ... => wszytsko 
print(tensor_two_d[..., 1])