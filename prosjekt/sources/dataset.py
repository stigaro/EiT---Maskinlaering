import os
import torch
import numpy as np

from PIL import Image

from sources.utility import load_json_file, coco_image_annotations_to_masks


class TacoDataset(object):
    def __init__(self, folder_path, transforms):
        # Define all the variables for the dataset
        self.folder_path = folder_path
        self.transforms = transforms
        self.information = load_json_file(folder_path + '/information.json')

        # Define all the image and annotation paths, sorting them to ensure that they are aligned
        self.images_folder = self.folder_path + '/images'
        self.image_paths = sorted([self.images_folder + '/' + image_file for image_file in os.listdir(self.images_folder)])
        self.annotations_folder = self.folder_path + '/annotations'
        self.annotation_paths = sorted([self.annotations_folder + '/' + annotation_file for annotation_file in os.listdir(self.annotations_folder)])

    def __getitem__(self, index):
        # Loads the image and the annotation
        image = Image.open(self.image_paths[index]).convert("RGB")
        image_annotation = load_json_file(self.annotation_paths[index])

        # Converts the annotation into nd-array of masks
        masks = coco_image_annotations_to_masks(image_annotation)

        # Extract the bounding boxes from the annotations,
        # because the COCO is stored in [x, y, width, height] format
        # we must convert it to [x_min, y_min, x_max, y_max] instead
        boxes = []
        for bounding_box in [dictionary['bbox'] for dictionary in image_annotation['annotations']]:
            [x_min, y_min, width, height] = bounding_box
            boxes.append([x_min, y_min, x_min + width, y_min + height])

        # Convert everything into a Torch Tensor
        image_id = torch.tensor([image_annotation['information']['id']])
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor([dictionary['category_id'] for dictionary in image_annotation['annotations']], dtype=torch.int64)
        areas = torch.as_tensor([dictionary['area'] for dictionary in image_annotation['annotations']], dtype=torch.float32)
        iscrowds = torch.as_tensor([dictionary['iscrowd'] for dictionary in image_annotation['annotations']], dtype=torch.bool)

        # Create the target dictionary
        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": image_id,
            "area": areas,
            "iscrowd": iscrowds
        }

        # Apply any transforms we have set
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        # Return the image and target
        return image, target

    def __len__(self):
        return len(self.image_paths)
