import requests
import json

taco = open("unprocessed/taco.json", "r")
taco = json.load(taco)

print(taco["images"])

# Data set is a giant dict with one key for images holding a list if dicts which
# again holds dicts with info and link to images. Images belongs to batches.