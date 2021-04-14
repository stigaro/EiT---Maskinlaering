from sources.constant import __SHAPE_BINARY, _BATCH_SIZE_BINARY
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models


def binary_model_builder(hp):
    model = models.Sequential()
    hp_filter_size1 = hp.Choice("filter_size1", values=[5,7])
    hp_stride1 = hp.Choice("stride_length1", values=(2,3))
    hp_conv1_filters = hp.Int('n_filters1', min_value=32, max_value=128, step=32)
    model.add(layers.Conv2D(hp_conv1_filters, (hp_filter_size1, hp_filter_size1),
                            strides=(hp_stride1, hp_stride1), activation='relu',
                            input_shape=__SHAPE_BINARY))
    hp_pooling1 = hp.Choice("pooling_size1", values=(2,3,4,5))
    model.add(layers.MaxPooling2D((hp_pooling1, hp_pooling1)))

    hp_conv2_filters = hp.Int('n_filters2', min_value=32, max_value=128, step=32)
    hp_filter_size2 = hp.Choice("filter_size2", values=[3, 5, 7])
    hp_stride2 = hp.Choice("stride_length2", values=(1, 2))
    model.add(layers.Conv2D(hp_conv2_filters, (hp_filter_size2, hp_filter_size2),
                            strides=(hp_stride2, hp_stride2),
                            activation='relu'))
    hp_pooling2 = hp.Choice("pooling_size2", values=(2, 3, 4))
    model.add(layers.MaxPooling2D((hp_pooling2, hp_pooling2)))

    """
    hp_bool_third_layer = hp.Choice("bool_third_conv_layer", values=[0,1])
    if (hp_bool_third_layer):
        hp_conv3_filters = hp.Int('n_filters3', min_value=64, max_value=128, step=32)
        hp_filter_size3 = hp.Choice("filter_size3", values=())
        hp_stride3 = hp.Choice("stride_length3", values=(1))
        print(model.summary())
        model.add(layers.Conv2D(hp_conv3_filters, (2, 2),
                                activation='relu'))
        print(model.summary())
    """


    # flatten layers
    model.add(layers.Flatten())

    # Tune the number of units in the first Dense layer
    # Choose an optimal value between 32-512
    hp_units1 = hp.Int('units', min_value=32, max_value=512, step=32)
    model.add(layers.Dense(units=hp_units1, activation='relu'))
    hp_dropout = hp.Float("dropout", min_value=0, max_value=0.3, step=0.05)
    model.add(layers.Dropout(hp_dropout))
    hp_units2 = hp.Int('units2', min_value=32, max_value=248, step=32)
    model.add(layers.Dense(hp_units2, activation='relu'))

    hp_bool_third_normal_layer = hp.Choice("bool_third_normal_layer", values=[0, 1])
    if(hp_bool_third_normal_layer):
        hp_units3 = hp.Int('units3', min_value=32, max_value=128, step=32)
        model.add(layers.Dense(hp_units3, activation='relu'))

    # layer for percent
    model.add(layers.Dense(2, activation='softmax'))

    # tune learning rate
    hp_learning_rate = hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')
    # compile model
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=hp_learning_rate),
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

    return model
