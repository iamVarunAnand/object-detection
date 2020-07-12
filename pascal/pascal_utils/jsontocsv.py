# import the necessary packages
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import json
import os


class JSONToCSV:
    def __init__(self, largest = True):
        # initialize the instance variables
        self.largest = largest

    def extract_annos(self, jfnames):
        # initialize a dictionary to store the annotations
        annos = {}

        # loop through the json files
        for jfname in jfnames:
            print(f"[INFO] extracting annotations from {jfname}")

            # read the json file
            jf = json.load(open(jfname, "r"))

            # extract the necessary data from the json file
            fnames = {img["id"]: img["file_name"] for img in jf["images"]}

            # calculate the jpeg images base directory from the json path
            base_dir = jfname.parents[0]

            # loop through the annotations
            for anno in jf["annotations"]:
                # extract the image id, label and bbox from the annotation
                img_id = anno["image_id"]
                lbl = anno["category_id"]
                bbox = anno["bbox"]

                # extract the appropriate filename
                fname = fnames[img_id]

                # build the complete file path
                fpath = Path(base_dir / "JPEGImages" / fname)

                # append the annotation to the dictionary
                L = annos.get(str(fpath), [])
                L.append((lbl, bbox))
                annos[str(fpath)] = L

        # return the extracted annotations
        return annos

    def get_largest(self, anno):
        # return the largest object in the annotation
        return sorted(anno, key = lambda x: x[1][2] * x[1][3], reverse = True)[0]

    def concat_convert(self, jfnames):
        # extract the annotations
        annos = self.extract_annos(jfnames)

        # check if only the largest annotation is to be retained
        if self.largest:
            # return the largest object in the annotation
            # loop through the annotations
            for key in annos.keys():
                # update the annotation with the largest object
                annos[key] = self.get_largest(annos[key])

        # initialize a dataframe to store the annotations
        df = pd.DataFrame(columns = ["img_paths", "annotations"])

        # loop through the annotations
        print("[INFO] converting annotations...")
        for i, (k, v) in tqdm(enumerate(annos.items()), total = len(annos)):
            df.loc[i] = [k, v]

        # sort the dataframe according to the image paths
        df.sort_values(axis = 0, by = "img_paths", inplace = True)

        # return the constructed dataframe
        return df
