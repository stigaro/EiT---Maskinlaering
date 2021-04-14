import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import kerastuner as kt
from sources.model_tuner import binary_model_builder
import os
from sources.dataset_binary import Loader
import numpy as np
from sources.constant import DEFAULT_IMAGE_SIZE


#os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

# import train images and labels
print("Importing train dataset")
dirpath= os.getcwd() + "/resources/dataset/trash_binary_numpy_dataset"
images_train = np.load(os.path.join(dirpath, "training_images.npy"))
label_train = np.load(os.path.join(dirpath, "training_labels.npy"))

# import test images and labels
print("Importing test dataset")
images_test = np.load(os.path.join(dirpath, "testing_images.npy"))
label_test = np.load(os.path.join(dirpath, "testing_labels.npy"))

tuner_dir = os.getcwd() + "/resources/models/binary_models"

tuner = kt.Hyperband(binary_model_builder,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3,
                     directory=tuner_dir,
                     project_name='kt_tuning',
                     distribution_strategy=tf.distribute.MirroredStrategy())

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

tuner.search(images_train, label_train, epochs=50, callbacks=[stop_early], use_multiprocessing=True,
             workers = 4, validation_split = 0.2)

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete.
The optimal number of filters for the first convolutional layer is {best_hps.get("n_filters1")}, while
optimal filter size is ({best_hps.get("filter_size1")}, {best_hps.get("filter_size1")}) and
optimal stride length is ({best_hps.get("stride_length1")}, {best_hps.get("stride_length1")}).
The optimal filter size for the first max pooling layer is ({best_hps.get("pooling_size1")},{best_hps.get("pooling_size1")}).
The optimal number of filters for the second convolutional layer is {best_hps.get("n_filters2")}, while
optimal filter size is ({best_hps.get("filter_size2")}, {best_hps.get("filter_size2")}) and
optimal stride length is ({best_hps.get("stride_length2")}, {best_hps.get("stride_length2")}).
The optimal filter size for the first max pooling layer is ({best_hps.get("pooling_size2")},{best_hps.get("pooling_size2")}).
The optimal number of units in the first densely-connected layer is {best_hps.get('units')}.
The optimal dropout rate for the drop out layer is {best_hps.get('dropout')}.
The optimal number of units in the second densely-connected layer is {best_hps.get('units2')}.
It is{"not" * best_hps.get("bool_third_normal_layer")} optimal to include a third densely-connected layer
 {("with " + str(best_hps.get("units3")) + " number of nodes.") * best_hps.get("bool_third_normal_layer")}
The optimal learning rate for the optimizer is {best_hps.get('learning_rate')}.
""")


# Build the model with the optimal hyperparameters and train it on the data for 10 epochs
model = tuner.hypermodel.build(best_hps)
model.summary()
# create path for saving model and checkpoints
checkpoint_path = os.getcwd() + "/resources/models/binary_models/tuned_model/checkpoints/cp-{epoch:03d}"
checkpoint_dir = os.path.dirname(checkpoint_path)
model_dir = os.path.dirname(checkpoint_dir)

# create callback object
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True)

history = model.fit(images_train, label_train, epochs=50,
                    callbacks=[cp_callback],
                    validation_split = 0.2)

val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))


# save the full fitted model from the best epoch
model.load_weights(os.path.join(checkpoint_dir, "cp-{:03d}".format(best_epoch)))

model.save(model_dir)
