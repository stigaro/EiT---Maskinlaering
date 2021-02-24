import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from math import sqrt
import numpy
import os
import random


class Viz():

    def __init__(self,
                outputs,
                targets,
                images,
                sq_num):

        self.outputs = outputs
        self.targets = targets
        self.images = images
        self.sq_num = sq_num

        self.pic_size = sqrt(self.sq_num)

        plt.figure(figsize=(10,10))


        for i in range(sq_num):
            ax = plt.subplot(self.pic_size,self.pic_size,i+1)
            plt.imshow(images[i])
            plt.title("T:" + "{:.2f}".format((targets[i])) \
                    + " O:" + "{:.2f}".format((outputs[i])))
            plt.axis("off")
            
        plt.show()
            



if __name__ == "__main__":
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
