import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt

from sources.constant import DEFAULT_IMAGE_SIZE, _BATCH_SIZE_BINARY
from sources.utility import normalize_rgb_image_with_label


class Loader:
    """
    A class for all loading dataset functions
    """

    @staticmethod
    def load_raw_dataset(dataset_path: str, image_shape = DEFAULT_IMAGE_SIZE, shuffle = True):
        """
        :param
            dataset_path: Path to where the dataset structure is located.
                Expect the path folder to have two subfolders; training and testing.
                Within each subfolder (training or testing) there should be a folder with the name of each class.
            image_shape: shape of all output images.
                Default is taken from constant DEFAULT_IMAGE_SIZE in sources.constant.py.
        :return:
            Tuple with training and testing datasets. Each dataset contains images and labels.
        """
        loaded_dataset_training = tf.keras.preprocessing.image_dataset_from_directory(
            pathlib.Path(dataset_path + "/training"),
            image_size=image_shape,
            class_names=["negative", "positive"],
            shuffle=shuffle,
            batch_size= _BATCH_SIZE_BINARY
        )
        normalized_dataset_training = loaded_dataset_training.map(normalize_rgb_image_with_label)
        loaded_dataset_testing = tf.keras.preprocessing.image_dataset_from_directory(
            pathlib.Path(dataset_path + "/testing"),
            image_size=image_shape,
            class_names=["negative", "positive"],
            shuffle=shuffle,
            batch_size=_BATCH_SIZE_BINARY
        )
        normalized_dataset_testing = loaded_dataset_testing.map(normalize_rgb_image_with_label)
        return normalized_dataset_training, normalized_dataset_testing

# TODO: Remove below when certain that it works
if __name__ == "__main__":
    dataset, dataset_test = Loader.load_raw_dataset(os.getcwd() + "/resources/dataset/trash_binary_dataset")
    print(dataset)
    for (image_batch, label_batch) in dataset:
        for (image, label) in zip(image_batch, label_batch):
            print(image.shape)
            print(image_batch.shape)
            print(label_batch.shape)
            plt.figure()
            plt.imshow(image)
            print(type(image))
            break
        break
    plt.show()