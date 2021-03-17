import os
import zipfile

if not os.getcwd().endswith("/prosjekt"):
    print("Script must be run from '/prosjekt' folder.")
    exit()

api_call = "kaggle datasets download -d"

# Read datasets from file
file = open("sources/scripts/downloading/datasets.txt", "r")
datasets = file.read()
datasets = datasets.split("\n")
file.close()

os.chdir("resources/dataset/unprocessed/.")

# Download and unzip datasets
for ds in datasets:
    os.system(api_call + " " + ds)
    downloaded_zip = ds.split("/")[1] + ".zip"
    print("Unzipping" + ds)
    zipfile.ZipFile(downloaded_zip).extractall()
    os.remove(downloaded_zip)

print("Completed")
