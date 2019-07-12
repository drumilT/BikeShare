import tensorflow as tf
from error_function import travel_data_error
tf.compat.v1.executing_eagerly()
c = tf.constant([1.0, 2.0 ,3.0, 4.0])
d = tf.constant([1.0, 2.0 ,3.0, 4.0])

val = tf.map_fn(lambda x: x , c)
val2 = travel_data_error(c,d)
print(val2)
print(val.numpy())