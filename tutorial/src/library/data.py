from tensorflow.keras import datasets
from matplotlib import pyplot as plt
import numpy as np


class Data:

    # Returns test images[1000][28][28], labels
    @staticmethod
    def get_training_data():
        training_data, _ = datasets.mnist.load_data()
        return training_data  # images, labels

    # Returns test images[1000][28][28], labels
    @staticmethod
    def get_test_data():
        _, test_data = datasets.mnist.load_data()
        return test_data

    
    # Return a single sample
    @staticmethod
    def get_single_sample():
        return Data.get_test_data()[0][0]

def shows_sample_data():

    images, labels = Data.get_training_data()

    plt.figure(figsize=(10, 10))

    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.xlabel(labels[i])
    plt.show()
