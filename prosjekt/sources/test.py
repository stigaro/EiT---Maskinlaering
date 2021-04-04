import tensorflow as tf
input_shape = (4, 7, 28, 28, 3)
x = tf.random.normal(input_shape)
y = tf.keras.layers.Conv2D(
4, 3, activation='relu', input_shape=input_shape[2:])(x)
print(y.shape)
