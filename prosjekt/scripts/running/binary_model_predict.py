import tensorflow as tf
import os
from sources.dataset_binary import Loader
from sources.visualization import VisualizerBinary
from sources.model import get_binary_model
import numpy as np

if __name__ == "__main__":
    # TODO: Check why code behaves weird for shuffle = TRUE
    """
    Code to test prediction of a model
    """
    # import test images and labels
    dirpath = os.getcwd() + "/resources/dataset/trash_binary_numpy_dataset"
    images_test = np.load(os.path.join(dirpath, "testing_images.npy"))
    label_test = np.load(os.path.join(dirpath, "testing_labels.npy"))

    # load model (model trained earlier)
    path_model = os.getcwd() + "/resources/models/binary_models/tuned_model"
    mymodel = tf.keras.models.load_model(path_model)

    # Loading of specific check point weights if wanted
    #mymodel.load_weights(os.getcwd() + "/resources/models/binary_models/checkpoints/cp-002")


    # evaluating the loaded model
    loss, acc = mymodel.evaluate(images_test, label_test, verbose=2)
    print('Restored model, test accuracy: {:5.2f}%'.format(100 * acc))
    pred = mymodel.predict(images_test)
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



    # create visualization class
    viz = VisualizerBinary(pred, images_test, label_test)
    # plot images
    viz.plot_images_and_pred(5,5, path_model)
    # plot ROC curve
    viz.plot_ROC(path_model)
