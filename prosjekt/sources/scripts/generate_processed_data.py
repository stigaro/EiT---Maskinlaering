import os
import json
import shutil
import random

__INPUT_PATH = 'resources/unprocessed_data'
__OUTPUT_PATH = 'resources/processed_data'
__WORKING_DIRECTORY = 'prosjekt'


def split_list(list, fraction):
    # Splits the dataset into two by the fraction
    split_index = int(fraction * len(list))
    return list[:split_index], list[split_index:]


def create_folder_and_copy_over_data(folder_name, file_names):
    # Generate the folders
    os.mkdir(__OUTPUT_PATH + '/' + folder_name)
    os.mkdir(__OUTPUT_PATH + '/' + folder_name + '/images')
    os.mkdir(__OUTPUT_PATH + '/' + folder_name + '/annotations')

    # Copy over the files
    for file_name in file_names:
        shutil.copyfile(__INPUT_PATH + '/images/' + file_name + '.jpg', __OUTPUT_PATH + '/' + folder_name + '/images/' + file_name + '.jpg')
        shutil.copyfile(__INPUT_PATH + '/annotations/' + file_name + '.json', __OUTPUT_PATH + '/' + folder_name + '/annotations/' + file_name + '.json')
        shutil.copyfile(__INPUT_PATH + '/information.json', __OUTPUT_PATH + '/' + folder_name + '/information.json')


# --------------------------------------------------------------------------- #
#                             PREPARATION CODE                                #
# --------------------------------------------------------------------------- #
# Checks for correct working directory before performing script
if not os.path.basename(os.getcwd()) == __WORKING_DIRECTORY:
    raise RuntimeError('Script must be run with working directory set at project folder root')

# Prepares data folder setup
shutil.rmtree(__OUTPUT_PATH, ignore_errors=True)
os.mkdir(__OUTPUT_PATH)


# --------------------------------------------------------------------------- #
#                                SCRIPT CODE                                  #
# --------------------------------------------------------------------------- #
# Loads all file names into lists, and shuffles it.
all_file_names = list(set([
    os.path.splitext(file_name)[0]
    for file_name
    in os.listdir('resources/unprocessed_data/images')
    if os.path.isfile('resources/unprocessed_data/images/' + file_name)
]))
random.shuffle(all_file_names)

# Splits into three separate file name lists
testing_file_names, training_file_names = split_list(all_file_names, 0.20)

# Copies from the images folder into the new directories
create_folder_and_copy_over_data('training', training_file_names)
create_folder_and_copy_over_data('testing', testing_file_names)
