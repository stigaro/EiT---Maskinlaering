import tensorflow as tf
import matplotlib.pyplot as plt
import random
from tensorflow.keras import datasets, layers, models
import numpy as np
from model import Model, plot_metrics
from viz import Viz

if __name__ == "__main__":

    # code to test viz and models classes together

    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    # choose a class, set its labels equal to 1 and all other classes equal to 0.
    for i in range(train_labels.size):
        if train_labels[i] == 6:
            train_labels[i] = 1
        else:
            train_labels[i] = 0

    for i in range(test_labels.size):
        if test_labels[i] == 6:
            test_labels[i] = 1
        else:
            test_labels[i] = 0

    # normalise
    train_images, test_images = train_images / 255.0, test_images / 255.0

    mymodel = Model(3, 0.15, (32, 32, 3))

    mymodel.compile(optimizer='sgd',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                    metrics=['accuracy'])

    history = mymodel.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))
    # mymodel.save()
    outputs = mymodel.call(test_images)

    sq_num = 36

    viz = Viz(outputs, test_labels, test_images, sq_num)

    plot_metrics(history)
