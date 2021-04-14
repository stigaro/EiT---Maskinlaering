from sources.constant import __SHAPE_BINARY, _BATCH_SIZE_BINARY
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models


def binary_model_builder(hp):
    model = models.Sequential()
    hp_filter_size = hp.Choice("filter_size", values=[5,7])
    hp_stride = hp.Choice("stride_length", values=(2,3))
    model.add(layers.Conv2D(_BATCH_SIZE_BINARY, (hp_filter_size, hp_filter_size),
                            strides=(hp_stride, hp_stride), activation='relu',
                            input_shape=__SHAPE_BINARY))
    model.add(layers.MaxPooling2D((3, 3)))

    model.add(layers.Conv2D(64, (5,5), activation='relu'))
    model.add(layers.MaxPooling2D((4, 4)))

    hp_bool_third_layer = hp.Choice("bool_third_conv_layer", values=[0,1])
    if (hp_bool_third_layer):
        model.add(layers.Conv2D(64, (2, 2), activation='relu'))

    # flatten layers
    model.add(layers.Flatten())

    # Tune the number of units in the first Dense layer
    # Choose an optimal value between 32-512
    hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
    model.add(layers.Dense(units=hp_units, activation='relu'))
    hp_dropout = hp.Float("dropout", min_value=0, max_value=0.3, step=0.05)
    model.add(layers.Dropout(hp_dropout))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(2, activation='softmax'))

    hp_learning_rate = hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')
    # compile model
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=hp_learning_rate),
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

    return model
