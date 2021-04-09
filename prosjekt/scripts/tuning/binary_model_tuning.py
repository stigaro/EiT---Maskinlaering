import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import kerastuner as kt
from sources.model_tuner import binary_model_builder
import os
from sources.dataset_binary import Loader


#os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

# import train and test dataset
dataset_train, dataset_test = Loader.load_raw_dataset(
    os.getcwd() + "/resources/dataset/trash_binary_dataset",
    shuffle=False
)

tuner_dir = os.getcwd() + "/resources/models/binary_models"

tuner = kt.Hyperband(binary_model_builder,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3,
                     directory=tuner_dir,
                     project_name='kt_tuning',
                     distribution_strategy=tf.distribute.MirroredStrategy())

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

tuner.search(dataset_train, epochs=50, callbacks=[stop_early], use_multiprocessing=True,
             workers = 4, validation_data=dataset_test)

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units')}, the optimal dropout rate is {best_hps.get('dropout')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")

# Build the model with the optimal hyperparameters and train it on the data for 10 epochs
model = tuner.hypermodel.build(best_hps)
history = model.fit(dataset_train, epochs=10)

val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

