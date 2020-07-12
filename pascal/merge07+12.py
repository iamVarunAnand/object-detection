# add the current directory to the path
import sys
sys.path.__add__(["."])

# import the necessary packages
from sklearn.model_selection import train_test_split
from pascal_utils.jsontocsv import JSONToCSV
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import argparse
import shutil
import json
import os

# construct an argument parser to parse the command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--seven", required = True, help = "path to directory containing VOC07 dataset")
ap.add_argument("-t", "--twelve", required = True, help = "path to directory containing VOC12 dataset")
ap.add_argument("-o", "--output", required = True, help = "path to store the merged dataset")
args = vars(ap.parse_args())

# define the required paths
VOC07 = Path(args["seven"])
VOC12 = Path(args["twelve"])
TRAIN07_JSON = Path(VOC07 / "train" / "train07.json")
VAL07_JSON = Path(VOC07 / "train" / "val07.json")
TEST07_JSON = Path(VOC07 / "test" / "test07.json")
JPEGS07 = Path(VOC07 / "train" / "JPEGImages")
TRAIN12_JSON = Path(VOC12 / "train" / "train12.json")
VAL12_JSON = Path(VOC12 / "train" / "val12.json")
JPEGS12 = Path(VOC12 / "train" / "JPEGImages")
OUTPUT_PATH = Path(args["output"])

# initialize the converter object
jtc = JSONToCSV(largest = True)

# concat the json files and convert the annotations into pandas dataframes
trainval_df = jtc.concat_convert([TRAIN07_JSON, VAL07_JSON, TRAIN12_JSON, VAL12_JSON])
test_df = jtc.concat_convert([TEST07_JSON])

# extract the labels from the trainval df
labels = [anno[0] for anno in trainval_df["annotations"]]

# split the trainval df
train_df, val_df = train_test_split(trainval_df, test_size = 0.2, random_state = 233, stratify = labels)

# print an update to the user
print(f"[INFO] number of training samples: {train_df.shape[0]}")
print(f"[INFO] number of validation samples: {val_df.shape[0]}")
print(f"[INFO] number of testing samples: {test_df.shape[0]}")

# create a list to store the different splits
dataset = [
    ("train", train_df),
    ("val", val_df),
    ("test", test_df)
]

# check if the output path exists, if not create it
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

# loop through the dataset, and copy the files to their respective folders
for split, df in dataset:
    print(f"[INFO] building {split} split...")

    # build the path to the current split
    SPLIT_PATH = Path(OUTPUT_PATH / split)

    # build the path to the images folder of the current split
    IMAGES_PATH = Path(SPLIT_PATH, "JPEGImages")

    # if the directory for the current split doesn't exist, create it
    if not os.path.exists(IMAGES_PATH):
        os.makedirs(IMAGES_PATH)

    # loop through the annotations and copy the appropriate files
    for i in tqdm(range(df.shape[0]), total = df.shape[0]):
        # extract the source path from the df
        src = df.iloc[i]["img_paths"]

        # extract the filename from the src path
        fname = src.split(os.path.sep)[-1]

        # build the destination path
        dest = Path(IMAGES_PATH / fname)

        # copy the file
        shutil.copy(src, dest)

        # modify the entry in the dataframe to contain only the filename
        df.iloc[i]["img_paths"] = fname

    # build the path to store the csv file
    CSV_PATH = Path(SPLIT_PATH / f"localisation_{split}.csv")

    # save the dataframe to disk
    df.to_csv(CSV_PATH, index = False)
