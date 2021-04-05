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

    # code to train model
    dataset_train, dataset_test = Loader.load_raw_dataset(
        os.getcwd() + "/resources/dataset/trash_binary_dataset"
    )
    mymodel = get_binary_model((256, 256, 3))

    mymodel.compile(optimizer='sgd',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                    metrics=['accuracy'])
    mymodel.summary()

    checkpoint_path = os.getcwd() + "/resources/models/binary_models/checkpoints/cp-{epoch:03d}"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=True)
    mymodel.save_weights(checkpoint_path.format(epoch=0))

    history = mymodel.fit(dataset_train, epochs=10,
                          validation_data=dataset_test,
                          callbacks=[cp_callback])
    mymodel.save(os.getcwd() + "/resources/models/binary_models/model1")

    # plot test and train accuracy for each epoch
    plot_metrics(history)
