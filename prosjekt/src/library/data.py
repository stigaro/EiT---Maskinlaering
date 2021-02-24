import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import pathlib

from src.library.constant import DEFAULT_IMAGE_SIZE
from src.library.utility import normalize_rgb_image


class DataBank:
    """
    TODO: lag
    """
    pass


class Loader:
    """
    TODO: gjer dette også
    """

    @staticmethod
    def load_raw_dataset(dataset_path: str, image_shape = DEFAULT_IMAGE_SIZE):
        """
        TODO: skriv inn javadoc
        :param dataset_path:
        :return:
        """
        loaded_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            pathlib.Path(dataset_path),
            image_size = image_shape
        )
        normalized_dataset = loaded_dataset.map(lambda image, label: (normalize_rgb_image(image), label))
        return normalized_dataset
