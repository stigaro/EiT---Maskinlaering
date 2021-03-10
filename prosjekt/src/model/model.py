import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models


class Model(tf.keras.Model):
        
    def __init__(self, dropout:float, shape:tuple):
        """
        Constructor for CNN model. Defines network structure.
            args:
                num_layers:int - number of convolutional and pooling layers
                dropout:float - dropout rate in FC layers
                shape:tuple - image shape [width, height, channels]
        """
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.shape = shape

        #max-pool filter size
        self.pool_size = (2,2)

        #Conv2d kernel size
        self.kernel_size = (3,3)


        self.model = models.Sequential()
        self.model.add(layers.Conv2D(self.shape[0], (3, 3), activation='relu', input_shape=self.shape))
        self.model.add(layers.MaxPooling2D((2, 2)))
    
        self.model.add(layers.Conv2D(self.shape[0]*2, self.kernel_size, activation='relu'))
        self.model.add(layers.MaxPooling2D(self.pool_size))
        
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(self.shape[0]*2, activation='relu'))
        #self.model.add(layers.Dropout(self.dropout))
        self.model.add(layers.Dense(32, activation='relu'))
        self.model.add(layers.Dense(16, activation='relu'))
        self.model.add(layers.Dense(2, activation='softmax'))
    
    def call(self, inputs):
        """ 
            Performs forward pass
            args:
                inputs - images
        """
        return self.model(inputs)


def plot_metrics(history):
    """
        Plots history metrics.
        args:
            history:dict - models output history
    """

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()


    
