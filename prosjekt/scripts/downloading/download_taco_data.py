import json
import os.path
import shutil
import sys
from io import BytesIO

import cv2
import numpy as np
import requests
from PIL import Image

from sources.utility import load_json_file, loading_bar, Constant

__TACO_DATASET_DICTIONARY_PATH = 'libraries/taco_master/data/annotations.json'
__OUTPUT_PATH = 'resources/unprocessed_data/trash_annotations_in_context'


# noinspection PyShadowingNames
def __construct_file_name(base_name, image_number, number_of_leading_zeroes):
    return base_name + '_{number}'.format(number=str(image_number).zfill(number_of_leading_zeroes)) + '.jpg'


# --------------------------------------------------------------------------- #
#                             PREPARATION CODE                                #
# --------------------------------------------------------------------------- #
# Checks for correct working directory before performing script
if not os.path.basename(os.getcwd()) == Constant.WORKING_DIRECTORY:
    raise RuntimeError('Script must be run with working directory set at project folder root')

# Prepares data folder setup
shutil.rmtree(__OUTPUT_PATH, ignore_errors=True)
os.mkdir(__OUTPUT_PATH)
os.mkdir(__OUTPUT_PATH + '/images')
os.mkdir(__OUTPUT_PATH + '/annotations')

# Loads the unprocessed_data dictionary defined by the TACO team.
dataset_dictionary = load_json_file(__TACO_DATASET_DICTIONARY_PATH)
number_of_images = len(dataset_dictionary['images'])

# Fixes the labels to work with Tensorflow (Cannot reserve 0)
for dictionary in dataset_dictionary['annotations']:
    dictionary['category_id'] += 1
for dictionary in dataset_dictionary['scene_annotations']:
    dictionary['background_ids'] = [background_id + 1 for background_id in dictionary['background_ids']]
for dictionary in dataset_dictionary['categories']:
    dictionary['id'] += 1
for dictionary in dataset_dictionary['scene_categories']:
    dictionary['id'] += 1


# --------------------------------------------------------------------------- #
#                          IMAGE DOWNLOADING CODE                             #
# --------------------------------------------------------------------------- #
# Update image names to be more fitting.
for image_number in range(number_of_images):
    # Because we are renaming we will fix the file_name of each image
    number_of_leading_zeroes = int(np.ceil(np.log10(number_of_images)))
    dataset_dictionary['images'][image_number]['file_name'] = __construct_file_name('image', image_number, number_of_leading_zeroes)

# Stores the unprocessed_data information again with updated image names for ease of access.
with open(__OUTPUT_PATH + '/information.json', 'w') as file:
    json.dump(dataset_dictionary, file)

# Runs iteratively to download and save all images
for image_number in range(number_of_images):

    # Retrieve the relevant information
    image_dictionary = dataset_dictionary['images'][image_number]

    # Load and save the TACO image, we will always use the largest version since this is the unprocessed data.
    image_file_path = os.path.join(__OUTPUT_PATH + '/images', image_dictionary['file_name'])
    response = requests.get(image_dictionary['flickr_url'])
    image = Image.open(BytesIO(response.content))
    if image._getexif():
        image.save(image_file_path, exif=image.info["exif"])
    else:
        image.save(image_file_path)

    # Sets up a loading bar to inform user of progress
    loading_bar(image_number, number_of_images)

sys.stdout.write('Finished\n')


# --------------------------------------------------------------------------- #
#                       ANNOTATIONS GENERATION CODE                           #
# --------------------------------------------------------------------------- #
# Initialize a datasets dictionary
data_dictionary = dict({})

# Fill in the dictionary with the information
for image_information in dataset_dictionary['images']:
    data_dictionary[image_information['id']] = dict({})
    data_dictionary[image_information['id']]['annotations'] = []
    data_dictionary[image_information['id']]['information'] = image_information
for annotation in dataset_dictionary['annotations']:
    data_dictionary[annotation['image_id']]['annotations'].append(annotation)

# Save each of the keys to file in the folder
for data in data_dictionary.values():
    image_name, image_extension = os.path.splitext(data['information']['file_name'])
    with open(__OUTPUT_PATH + '/annotations/' + image_name + '.json', 'w') as file:
        json.dump(data, file)
