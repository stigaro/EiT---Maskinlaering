import tensorflow as tf
import os
from sources.dataset_binary import Loader
from sources.visualization import VisualizerBinary

if __name__ == "__main__":

    # code to test prediction of a model
    dataset_train, dataset_test = Loader.load_raw_dataset(
        os.getcwd() + "/resources/dataset/trash_binary_dataset"
    )

    mymodel = tf.keras.models.load_model(os.getcwd() + "/resources/models/binary_models/model1")
    pred = mymodel.predict(dataset_test)
    print(pred)

    viz = VisualizerBinary(pred, dataset_test)
    viz.plot_images_and_pred(5, 5)