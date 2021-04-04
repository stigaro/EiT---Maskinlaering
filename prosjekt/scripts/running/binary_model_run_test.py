import tensorflow as tf
import matplotlib.pyplot as plt
import random
from tensorflow.keras import datasets, layers, models
import numpy as np
from sources.model import BasicBinaryModel, plot_metrics
from sources.visualization import Visualizer
import os
from sources.dataset_binary import Loader

if __name__ == "__main__":

    # code to test viz and models classes together
    dataset_train, dataset_test = Loader.load_raw_dataset(
        os.getcwd() + "/resources/dataset/trash_binary_dataset"
    )
    print("aslkdjals")
    mymodel = BasicBinaryModel(0.15, (256, 256, 3))

    print("kjalsdjlsa")
    mymodel.compile(optimizer='sgd',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                    metrics=['accuracy'])

    print("hei")
    history = mymodel.fit(dataset_train, validation_data=dataset_test, validation_steps=2)
    # mymodel.save()
    outputs = mymodel.call(test_images)

    sq_num = 36

    viz = Visualizer(outputs, test_labels, test_images, sq_num)

    plot_metrics(history)
