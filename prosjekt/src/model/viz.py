import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.metrics import roc_curve, roc_auc_score
import tensorflow as tf
from math import sqrt
import numpy as np
import os
import random


class Viz():

    def __init__(self,
                outputs,
                targets,
                images,
                sq_num):

        """
            Constructor for visualisation class.
            args:
                outputs:tensor - predictions from model
                targets:tensor - data labels
                images:tensor - images to be visualised
                sq_num:int - how many pictures to plot, must be a square number

        """

        self.outputs = [np.argmax(output) for output in outputs]
        self.targets = targets
        self.images = images
        self.sq_num = sq_num

        self.pic_size = sqrt(self.sq_num)

        plt.figure(figsize=(10,10))


        for i in range(sq_num):
            ax = plt.subplot(self.pic_size,self.pic_size,i+1)
            plt.imshow(images[i])
            plt.title("T: " + str(self.targets[i]) + " O: " + str(self.outputs[i]))
            plt.axis("off")
            
        plt.show()

    def plot_ROC(self):
        
        """
            Plots ROC curve based on targets and outputs.
        """

        fpr, tpr, thresholds = roc_curve(self.targets, self.outputs)
        plt.plot(fpr,tpr)
        plt.axis([0,1,0,1])
        plt.xlabel('FP Rate')
        plt.ylabel('TP Rate')
        plt.show()

    def get_AUC(self):
        """
            Returns AUC score based on targets and outputs.
        """
        return roc_auc_score(self.targets, self.outputs)



if __name__ == "__main__":
    #test code for this class
    images = []
    outputs = []
    targets = []

    folder = "Pics"
    for fname in os.listdir(folder):
        img = mpimg.imread(os.path.join(folder, fname))
        images.append(img)
    
    #make fake outputs and targets
    outputs = [None] * len(images)
    targets = [None] * len(images)
    
    for i in range(len(images)):
        outputs[i] = random.random()
        targets[i] = random.randint(0,1)
        
    sq_num = 9
    viz = Viz(outputs, targets, images, sq_num)
    viz.plot_ROC()
