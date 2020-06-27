# add the project directory to python path
import sys
sys.path.__add__(["."])

# force tensorflow to use CPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# import the necessary packages
from tensorflow.keras.applications.resnet import ResNet50, preprocess_input
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from imutils.object_detection import non_max_suppression
from object_detection.helpers import sliding_window
from object_detection.helpers import pyramid
from cv2 import cv2
import numpy as np
import argparse
import imutils
import time

# construct an argument parser to parse the command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "path to input image")
ap.add_argument("-s", "--size", type = str, default = "(200, 150)", help = "ROI size (in pixels)")
ap.add_argument("-c", "--min_conf", type = float, default = 0.9, help = "minimum probability to filter weak detections")
ap.add_argument("-v", "--visualize", type = int, default = -1,
                help = "whether or not to show extra visualizations for debugging")
args = vars(ap.parse_args())

# initialize some useful constants
WIDTH = 600
PYR_SCALE = 1.5
WIN_STEP = 16
ROI_SIZE = eval(args["size"])
INPUT_SIZE = (224, 224)

# read the input image and resize it to the appropriate size
orig = cv2.imread(args["image"])
orig = imutils.resize(orig, width = WIDTH)
(H, W) = orig.shape[:2]

# initialize the image pyramid
pyr = pyramid(orig, scale = PYR_SCALE, min_size = ROI_SIZE)

# initialize lists to hold the ROIs and their locations
rois = []
locs = []

# time how long it takes to generate the rois
start = time.time()

# loop through the pyramids and slide a window through the images
for img in pyr:
    # calculate the scale (to upsample bbox locations)
    scale = W / float(img.shape[1])

    # slide the window through the current img
    for (x, y, orig_roi) in sliding_window(img, WIN_STEP, ROI_SIZE):
        # calculate the scaled dimensions
        x = int(x * scale)
        y = int(y * scale)
        w = int(ROI_SIZE[0] * scale)
        h = int(ROI_SIZE[1] * scale)

        # preprocess the ROI
        roi = cv2.resize(orig_roi, INPUT_SIZE)
        roi = img_to_array(roi)
        roi = preprocess_input(roi)

        # update the ROI list along with their correspoding coordinates
        rois.append(roi)
        locs.append((x, y, x + w, y + h))

        # check if the sliding process is to be visualized
        if args["visualize"] > 0:
            # clone the original image
            clone = orig.copy()
            cv2.rectangle(clone, (x, y), ((x + w), (y + h)), (0, 0, 255), 2)

            # show the visualization and the current ROI
            cv2.imshow("Visualization", clone)
            cv2.imshow("ROI", roi)
            cv2.waitKey(0)

# calculate how long the sliding window process took
end = time.time()
print(f"[INFO] looping over the pyramid took {(end - start):.5f} seconds")

# convert the rois to a numpy array
rois = np.array(rois)

# initialize the pretrained model
model = ResNet50(weights = 'imagenet', input_shape = (224, 224, 3))

# classify each of the rois and calculate how long the process took
print("[INFO] classifying ROIs...")
start = time.time()
preds = model.predict(rois)
end = time.time()
print(f"[INFO] classifying ROIs took {(end - start):.5f} seconds")

# decode the predictions
preds = decode_predictions(preds, top = 1)

# initialize a dictionary that maps class -> ROIs
labels = {}

# loop through the predictions
for i, pred in enumerate(preds):
    # grab the prediction information for the current ROI
    (synset_id, label, prob) = pred[0]

    # filter out the weak predictions
    if prob > args["min_conf"]:
        # grab the corresponding bounding box
        bbox = locs[i]

        # add this bbox to the list of predictions for the current label
        L = labels.get(label, [])
        L.append((bbox, prob))
        labels[label] = L

# loop over the labels for each of the detected objects in the image
for label in labels.keys():
    # clone the original image (to draw on it)
    print(f"[INFO] showing results for {label}")
    clone = orig.copy()

    # loop over the bounding boxes for the current label
    for (bbox, prob) in labels[label]:
        # draw the bounding box
        (startx, starty, endx, endy) = bbox
        cv2.rectangle(clone, (startx, starty), (endx, endy), (0, 0, 255), 2)

    # show the results before applying non max suppresion
    cv2.imshow("Before Non-Max", clone)
    cv2.waitKey(0)

    # extract the bounding boxes and class probabilites and apply non max supression
    boxes = np.array([p[0] for p in labels[label]])
    probs = np.array([p[1] for p in labels[label]])
    boxes = non_max_suppression(boxes, probs)

    # clone the original image again and loop over all the remaining boxes
    clone = orig.copy()
    for (startx, starty, endx, endy) in boxes:
        # draw the bounding box
        cv2.rectangle(clone, (startx, starty), (endx, endy), (0, 0, 255), 2)

        # write the class label
        y = starty - 10 if starty - 10 > 10 else starty + 10
        cv2.putText(clone, label, (startx, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    # show the results after non max suppresion
    cv2.imshow("After Non-Max", clone)
    cv2.waitKey(0)

# close all the windows
cv2.destroyAllWindows()
