import tensorflow as tf
import matplotlib.pyplot as plt
import random
from tensorflow.keras import datasets, layers, models
import numpy as np
from sources.model import get_binary_model, plot_metrics
from sources.visualization import Visualizer
import os
from sources.dataset_binary import Loader

if __name__ == "__main__":

    # code to train model
    dataset_train, dataset_test = Loader.load_raw_dataset(
        os.getcwd() + "/resources/dataset/trash_binary_dataset"
    )
    mymodel = get_binary_model((256, 256, 3))

    mymodel.compile(optimizer='sgd',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                    metrics=['accuracy'])
    mymodel.summary()
    history = mymodel.fit(dataset_train, epochs=3, steps_per_epoch =20,
                          validation_data = dataset_test, shuffle = False)
    mymodel.save(os.getcwd() + "/resources/models/binary_models/model1")

    print(history.history)
    plot_metrics(history)
