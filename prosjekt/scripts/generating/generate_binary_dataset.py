import os
import json
import pathlib
import shutil
import random
import glob

from sources.utility import Constant

__INPUT_PATH_POSITIVE = [
    'resources/dataset/unprocessed/trash_annotations_in_context'
]
__INPUT_PATH_NEGATIVE = [
    'resources/dataset/unprocessed/flowers',
    'resources/dataset/unprocessed_data/natural_images'
]
__OUTPUT_PATH = 'resources/datasets/trash_binary_dataset'
__TRAIN_TEST_SPLIT = 0.20


def split_list(list, fraction):
    # Splits the datasets into two by the fraction
    split_index = int(fraction * len(list))
    return list[:split_index], list[split_index:]


def create_folder_and_copy_over_data(folder_name, positive_file_paths, negative_file_paths):
    # Generate the folders
    os.mkdir(__OUTPUT_PATH + '/' + folder_name)
    os.mkdir(__OUTPUT_PATH + '/' + folder_name + '/positive')
    os.mkdir(__OUTPUT_PATH + '/' + folder_name + '/negative')

    # Copy over the positive files
    for file_path in positive_file_paths:
        shutil.copyfile(file_path.__str__(), __OUTPUT_PATH + '/' + folder_name + '/positive/' + file_path.name)

    # Copy over the negative files
    for file_path in negative_file_paths:
        shutil.copyfile(file_path.__str__(), __OUTPUT_PATH + '/' + folder_name + '/negative/' + file_path.name)

# --------------------------------------------------------------------------- #
#                             PREPARATION CODE                                #
# --------------------------------------------------------------------------- #
# Checks for correct working directory before performing script
if not os.path.basename(os.getcwd()) == Constant.WORKING_DIRECTORY:
    raise RuntimeError('Script must be run with working directory set at project folder root')

# Prepares data folder setup
shutil.rmtree(__OUTPUT_PATH, ignore_errors=True)
os.mkdir(__OUTPUT_PATH)


# --------------------------------------------------------------------------- #
#                                SCRIPT CODE                                  #
# --------------------------------------------------------------------------- #
# Loads all of the files to be used
all_positive_file_paths = []
for positive_path in __INPUT_PATH_POSITIVE:
    all_positive_file_paths.extend([path.relative_to('') for path in pathlib.Path(positive_path).rglob('*.jpg')])
all_negative_file_paths = []
for negative_path in __INPUT_PATH_NEGATIVE:
    all_negative_file_paths.extend([path.relative_to('') for path in pathlib.Path(negative_path).rglob('*.jpg')])

# Shuffles the data
random.shuffle(all_positive_file_paths)
random.shuffle(all_negative_file_paths)

# Ensure class balance by only using as many negative as positive
# TODO: Make this step unnecessary by weighting the importance of each model input. Må også endre for følgefeil i koden
all_negative_file_paths = all_negative_file_paths[:len(all_positive_file_paths)]

# Splits into testing and training splits
positive_testing, positive_training = split_list(all_positive_file_paths, __TRAIN_TEST_SPLIT)
negative_testing, negative_training = split_list(all_negative_file_paths, __TRAIN_TEST_SPLIT)

# Copies from the images folder into the new directories
create_folder_and_copy_over_data('training', positive_training, negative_training)
create_folder_and_copy_over_data('testing', positive_testing, negative_testing)
