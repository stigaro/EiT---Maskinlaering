# EiT - Maskinlæring
Repository for EiT gruppe 05

## Downloading datasets

### TACO dataset

Run this command from root folder to download dataset images with default parameters:

```
python3 prosjekt/src/resource/dataset/taco_downloader.py
```

To edit dataset target use:

```
--dataset_path "path/to/dataset"
```

To edit download destination use:

```
--images_destination "path/to/download/destination"
```

### Kaggle datasets

Download the Kaggle API:

```
pip3 install kaggle
```

Ensure API credentials are correct, read under 'API Credentials':

https://github.com/Kaggle/kaggle-api

From \prosjekt run:

```
python3 sources/scripts/downloading/kaggle_downloader.py
```