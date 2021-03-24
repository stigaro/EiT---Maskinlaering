import os
import zipfile

from sources.utility import Constant

# Checks for correct working directory before performing script
if not os.path.basename(os.getcwd()) == Constant.WORKING_DIRECTORY:
    raise RuntimeError('Script must be run with working directory set at project folder root')

api_call = "kaggle datasets download -d"

# Read the datasets to load from file
with open("scripts/downloading/kaggle_datasets.txt", "r") as file:
    datasets = file.read().split("\n")

os.chdir("resources/unprocessed_data/.")

# Download and unzip datasets
for ds in datasets:
    os.system(api_call + " " + ds)
    downloaded_zip = ds.split("/")[1] + ".zip"
    print("Unzipping" + ds)
    zipfile.ZipFile(downloaded_zip).extractall()
    os.remove(downloaded_zip)

print("Completed")
