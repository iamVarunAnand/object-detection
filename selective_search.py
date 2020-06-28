# force tensorflow to use the CPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# import the necessary packages
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications.resnet import ResNet50, preprocess_input
from imutils.object_detection import non_max_suppression
from cv2 import cv2
import numpy as np
import argparse
import imutils
import time

# construct an argument parser to parse the command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "path to input image")
ap.add_argument("-m", "--mode", choices = ["fast", "slow"], required = True, help = "selective search mode")
ap.add_argument("-c", "--min_conf", type = float, default = 0.9, help = "threshold to filter out weak predictions")
args = vars(ap.parse_args())

# initialize necessary constants
WIDTH = 600
INPUT_SHAPE = (224, 224)
BS = 128

# spped up opencv using multiple threads
cv2.setUseOptimized(True)
cv2.setNumThreads(4)

# load the image and resize it to the appropriate dimensions
img = cv2.imread(args["image"])
img = imutils.resize(img, width = WIDTH)

# initialize the selective search object using the default parameters and set the input image
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(img)

# switch to the appropriate mode
if args["mode"] == "fast":
    ss.switchToSelectiveSearchFast()
else:
    ss.switchToSelectiveSearchQuality()

# get the region proposals
start = time.time()
rects = ss.process()
end = time.time()
print(f"[INFO] total number of region proposals = {len(rects)}")
print(f"[INFO] selective search took {(end - start):0.5f} seconds")

# initialize a list to store the regions of interest
rois = []
locs = []

# loop through the region proposals
for rect in rects:
    # extract the ROI coordinates
    x, y, w, h = rect

    # extract the region of interest from the image
    roi = img[y: y + h, x: x + w]

    # preprocess the ROI
    roi = cv2.resize(roi, INPUT_SHAPE, interpolation = cv2.INTER_AREA)
    roi = img_to_array(roi)
    roi = preprocess_input(roi)

    # append the current ROI to the ROIs list
    rois.append(roi)

    # modify the bbox coordinates and append them to the locs list
    locs.append((x, y, x + w, y + h))

# convert the ROI list to a numpy array
rois = np.array(rois)

# load the pretrained model
model = ResNet50(weights = "imagenet", input_shape = (224, 224, 3))

# get the predictions
print("[INFO] classifying the region proposals...")
start = time.time()
preds = model.predict(rois, batch_size = BS)
end = time.time()
print(f"[INFO] classification took {(end - start):0.5f} seconds")

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
    clone = img.copy()

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
    clone = img.copy()
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
