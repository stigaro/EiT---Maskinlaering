from tensorflow.keras import models, layers, losses
from src.library.data import *

sample = Data.get_single_sample()
# shape = sample.shape[0], sample.shape[1], 1
shape = (28,28,1)


model = models.Sequential()

# Number of general filters, size of filters, relu activation function
model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=shape))

# Pooling with 2x2 window
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(10))

def compile_model():
    model.compile(
        optimizer="adam",
        loss=losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

def get_loss_acc():
    # loss, acc
    return model.evaluate(test_images,  test_labels, verbose=2)