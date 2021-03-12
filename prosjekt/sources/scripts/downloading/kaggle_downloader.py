import os
import zipfile

if not os.getcwd().endswith("/prosjekt"):
    print("Script must be run from '/prosjekt' folder.")
    exit()

api_call = "kaggle datasets download -d"

# TODO lese fra fil
datasets = [
    "prasunroy/natural-images",
    "alxmamaev/flowers-recognition",
    "paramaggarwal/fashion-product-images-dataset",
    "arnaud58/landscape-pictures",
    "",
]

os.chdir("resources/dataset/unprocessed/.")

for ds in datasets:
    os.system(api_call + " " + ds)
    downloaded_zip = ds.split("/")[1] + ".zip"
    zipfile.ZipFile(downloaded_zip).extractall()
    os.remove(downloaded_zip)

print("Completed")
