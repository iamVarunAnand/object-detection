# import the necessary packages
from cv2 import cv2
import argparse
import imutils

# construct an argument parser to parse the command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "path to input image")
ap.add_argument("-m", "--mode", choices = ["fast", "slow"], required = True, help = "selective search mode")
args = vars(ap.parse_args())

# initialize necessary constants
WIDTH = 600
MAX_RECTS = 250

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
rects = ss.process()
print(f"[INFO] total number of region proposals = {len(rects)}")
rects = rects[:MAX_RECTS]

# create a copy of the image to plot the region proposals
clone = img.copy()

# iterate over all the region proposals
for i, rect in enumerate(rects):
    # draw a rectangle for the current region proposal
    x, y, w, h = rect
    cv2.rectangle(clone, (x, y), (x + w, y + h), (0, 0, 255), 1, cv2.LINE_AA)

# show the image
cv2.imshow("Output", clone)
cv2.waitKey(0)
cv2.destroyAllWindows()
