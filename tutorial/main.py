from src.library.data import *
from src.model.cnn.cnn import *
import numpy as np
import matplotlib.pyplot as plt

shows_sample_data()

print(Data.get_single_sample().shape)

compile_model()

train_images, train_labels = Data.get_training_data()
test_images, test_labels = Data.get_test_data()

history = model.fit(
    train_images,
    train_labels,
    epochs=10,
    validation_data=(test_images, test_labels),
)

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

print(get_loss_acc()[1])