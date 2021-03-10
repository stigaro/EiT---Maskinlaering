# EiT - Maskinlæring
Repository for EiT gruppe 05

## Download taco dataset

Run this command from root folder to download dataset images with default parameters:

```
python3 prosjekt/src/scripts/taco_downloader.py
```

To edit dataset target use:

```
--dataset_path "path/to/dataset"
```

To edit download destination use:

```
--images_destination "path/to/download/destination"
```

## Copy taco dataset files for binary trash classification

Run this command from root folder to copy dataset images from a subfolder
in unprocessed to a subfolder in processed:

```
python3 prosjekt/src/scripts/taco_downloader.py
```
These copied images will be used for training a binary trash classifier.