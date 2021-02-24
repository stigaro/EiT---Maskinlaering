import tensorflow as tf

normalize_rgb_image = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
