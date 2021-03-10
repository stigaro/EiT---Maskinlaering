"""
This script copies all taco image files into src/resource/dataset/processed/binary_trash_dataset/trash folder.
If the folder does not already exist it is created. Each images retains it's original name,
but is given a prefix "batch" + str(batch_number) + "_".
"""

import shutil
import os

source = "src/resource/dataset/unprocessed/taco/images"
dest1 = "src/resource/dataset/processed/binary_trash_dataset/trash"

if not os.path.isdir(dest1):
    os.mkdir(dest1)

subfolders = os.listdir(source)

batch = 0
for sf in subfolders:
    batch += 1
    files = os.listdir(source + "/" + sf)
    for f in files:
        shutil.copy(source + "/" + sf + "/" + f, dest1 + "/batch" + str(batch) + "_" + f)
