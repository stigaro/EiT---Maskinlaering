from src.library.data import *
import matplotlib.pyplot as plt

dataset = Loader.load_raw_dataset("C:/Users/Torbjørn/Git/EiT---Maskinlaering/prosjekt/src/resource/dataset/unprocessed/mock")
print(dataset)
for (image_batch, label_batch) in dataset:
    for (image, label) in zip(image_batch, label_batch):
        print(image.shape)
        plt.figure()
        plt.imshow(image)
        print(image)
        print(type(image))
        break
plt.show()