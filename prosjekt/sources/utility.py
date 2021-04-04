import json
import sys
import numpy as np
import pycocotools.mask as coco_mask_utility


def loading_bar(number, total_number):
    x = int(Constant.LOADING_BAR_SIZE * number / total_number)
    sys.stdout.write("%s[%s%s] - %i/%i\r" % ('Loading: ', "=" * x, "." * (Constant.LOADING_BAR_SIZE - x),
                                             number, total_number))
    sys.stdout.flush()


def load_json_file(json_file_path):
    with open(json_file_path, 'r') as json_file:
        return json.loads(json_file.read())


def coco_image_annotations_to_masks(image_annotation):
    """
    Helper function to turn a COCO image annotation into a masks array.
    (Interpreted 'image annotation' as the custom dictionary we construct when generating processed data folder)
    """
    image_information = image_annotation['information']

    image_masks = []
    for annotation in image_annotation['annotations']:
        all_rle = coco_mask_utility.frPyObjects(annotation['segmentation'], image_information['height'], image_information['width'])
        rle = coco_mask_utility.merge(all_rle)
        image_masks.append(coco_mask_utility.decode(rle))

    return image_masks


class Constant:
    WORKING_DIRECTORY = 'prosjekt'
    LOADING_BAR_SIZE = 30
