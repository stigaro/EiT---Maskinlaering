import os
from sources.dataset_binary import Loader
import numpy as np
from sources.constant import DEFAULT_IMAGE_SIZE
import tensorflow as tf

# import train and test dataset
dataset_train, dataset_test = Loader.load_raw_dataset(
    os.getcwd() + "/resources/dataset/trash_binary_dataset",
    shuffle=False
)

# convert train dataset to a numpy array

# find number of images in test dataset
n = 0
for (image_batch, label_batch) in dataset_train:
    batch_size = label_batch.shape[0]
    n += batch_size
    print(n, "images fetched from map dataset")

# create empty numpy arrays to store images and labels
label_train = np.empty(n)
dataset_train_np = np.empty((n, DEFAULT_IMAGE_SIZE[0], DEFAULT_IMAGE_SIZE[1], 3))
# fetch images from map dataset into n
k = 0
for (image_batch, label_batch) in dataset_train:
    batch_size = label_batch.shape[0]
    label_train[k:k + batch_size] = label_batch
    dataset_train_np[k:k + batch_size, :, :, :] = image_batch
    k += batch_size
    print(k, "out of", n)

print("Saving training data in numpy format")

path_training_images = os.getcwd() + "/resources/dataset/trash_binary_numpy_dataset/training_images.npy"
if not os.path.exists(os.path.dirname(path_training_images)):
    os.makedirs(os.path.dirname(path_training_images))
np.save(path_training_images, dataset_train_np)
print("Binary training images saved to path", path_training_images)

path_training_labels = os.getcwd() + "/resources/dataset/trash_binary_numpy_dataset/training_labels.npy"
if not os.path.exists(os.path.dirname(path_training_labels)):
    os.makedirs(os.path.dirname(path_training_labels))
np.save(path_training_labels, label_train)
print("Binary training labels saved to path", path_training_labels)


# convert test dataset to a numpy array

# find number of images in test dataset
n = 0
for (image_batch, label_batch) in dataset_test:
    batch_size = label_batch.shape[0]
    n += batch_size
    print(n, "images counted from map dataset")

label_test = np.empty(n)
dataset_test_np = np.empty((n, DEFAULT_IMAGE_SIZE[0], DEFAULT_IMAGE_SIZE[1], 3))
k = 0
for (image_batch, label_batch) in dataset_test:
    batch_size = label_batch.shape[0]
    label_test[k:k + batch_size] = label_batch
    dataset_test_np[k:k + batch_size, :, :, :] = image_batch
    k += batch_size
    print(k, "out of", n)

print("Saving testing data in numpy format")

path_testing_images = os.getcwd() + "/resources/dataset/trash_binary_numpy_dataset/testing_images.npy"
if not os.path.exists(os.path.dirname(path_testing_images)):
    os.makedirs(os.path.dirname(path_testing_images))
np.save(path_testing_images, dataset_test_np)
print("Binary testing images saved to path", path_testing_images)

path_testing_labels = os.getcwd() + "/resources/dataset/trash_binary_numpy_dataset/testing_labels.npy"
if not os.path.exists(os.path.dirname(path_testing_labels)):
    os.makedirs(os.path.dirname(path_testing_labels))
np.save(path_testing_labels, label_test)
print("Binary testing labels saved to path", path_testing_labels)