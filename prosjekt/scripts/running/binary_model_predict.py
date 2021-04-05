import tensorflow as tf
import os
from sources.dataset_binary import Loader
from sources.visualization import VisualizerBinary
from sources.model import get_binary_model
import numpy as np

if __name__ == "__main__":
    # TODO: Check why code behaves weird for shuffle = TRUE
    # code to test prediction of a model
    dataset_train, dataset_test = Loader.load_raw_dataset(
        os.getcwd() + "/resources/dataset/trash_binary_dataset",
        shuffle=False
    )

    mymodel = tf.keras.models.load_model(os.getcwd() + "/resources/models/binary_models/model1")
    loss, acc = mymodel.evaluate(dataset_test, verbose=2)
    print('Restored model, test accuracy: {:5.2f}%'.format(100 * acc))
    pred = mymodel.predict(dataset_test)
    # Below is code for testing why shuffle = True gives weird result
    # print(pred)
    # print([np.argmax(p) for p in pred])
    # true_label = np.empty(len(pred[:, 1]))
    # k = 0
    # for (image_batch, label_batch) in dataset_test:
    #     batch_size = label_batch.shape[0]
    #     true_label[k:k + batch_size] = label_batch
    #     k += batch_size
    # print(true_label)

    viz = VisualizerBinary(pred, dataset_test)
    viz.plot_images_and_pred(5,5)
    viz.plot_ROC()
