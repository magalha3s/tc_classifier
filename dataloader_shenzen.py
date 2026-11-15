import pickle
from pathlib import Path

SPLITS_PATH = None
CSV_PATH = None

with open(SPLITS_PATH, "rb") as splits_file:
    splits = pickle.load(splits_file)