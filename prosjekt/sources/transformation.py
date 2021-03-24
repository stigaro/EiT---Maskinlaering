import random
import torch
import numpy as np
from PIL import Image

from torchvision.transforms import functional


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class Resize(object):
    def __init__(self, resize_shape):
        self.resize_shape = resize_shape

    def __call__(self, image, target):
        # Calculate the scales we need to apply
        width_scaling = self.resize_shape[0] / image.size[0]
        height_scaling = self.resize_shape[1] / image.size[1]

        # Rescales the boxes tensor
        target['boxes'] = torch.as_tensor(list(map(
            lambda box: [box[0] * width_scaling, box[1] * height_scaling, box[2] * width_scaling, box[3] * height_scaling],
            target['boxes'].tolist())))

        # Rescales the masks tensor
        target['masks'] = torch.as_tensor(list(map(
            lambda mask: np.array(Image.fromarray(mask.numpy()).resize(self.resize_shape)),
            target['masks'])))

        # TODO: We do not modify 'area' as its not directly used. This should be remedied some time

        return image.resize(self.resize_shape), target


class ToTensor(object):
    def __call__(self, image, target):
        image = functional.to_tensor(image)
        return image, target
