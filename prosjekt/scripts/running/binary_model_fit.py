import tensorflow as tf
import matplotlib.pyplot as plt
import random
from tensorflow.keras import datasets, layers, models
import numpy as np
from sources.model import get_binary_model, plot_metrics
from sources.visualization import Visualizer
import os
from sources.dataset_binary import Loader
from sources.visualization import VisualizerBinary

if __name__ == "__main__":
    """
    Code to train model and save checkpoints as well as finished model
    """
    # import train images and labels
    dirpath= os.getcwd() + "/resources/dataset/trash_binary_numpy_dataset"
    images_train = np.load(os.path.join(dirpath, "training_images.npy"))
    label_train = np.load(os.path.join(dirpath, "training_labels.npy"))

    # import test images and labels
    images_test = np.load(os.path.join(dirpath, "testing_images.npy"))
    label_test = np.load(os.path.join(dirpath, "testing_labels.npy"))

    # convert from numpy arrays to tensor
    images_train = tf.convert_to_tensor(images_train)
    label_train = tf.convert_to_tensor(label_train)
    images_test = tf.convert_to_tensor(images_test)
    label_test = tf.convert_to_tensor(label_test)

    # create new model
    mymodel = get_binary_model((256, 256, 3))

    # compile model
    mymodel.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                    metrics=['accuracy'])

    mymodel.summary() # print model structure

    # create path for saving checkpoints
    checkpoint_path = os.getcwd() + "/resources/models/binary_models/checkpoints/cp-{epoch:03d}"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # create callback object
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=True)

    # save weights from before first epoch. Might be uneccecary
    mymodel.save_weights(checkpoint_path.format(epoch=0))

    # fit the model and save the model weights after each epoch
    history = mymodel.fit(images_train, label_train, epochs=10,
                          callbacks=[cp_callback], validation_data=(images_test, label_test))
    # save the full fitted model
    mymodel.save(os.getcwd() + "/resources/models/binary_models/model1")

    # plot test and train accuracy for each epoch
    plot_metrics(history)

