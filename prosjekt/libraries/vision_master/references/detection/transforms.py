import random
import torch
import numpy as np
from PIL import Image

from torchvision.transforms import functional as F


def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints(keypoints, width)
                target["keypoints"] = keypoints
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
        image = F.to_tensor(image)
        return image, target
