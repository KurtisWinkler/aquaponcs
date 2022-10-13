import cv2 as cv
import numpy as np

# Read image
im = cv.imread("ex11.tif")

im_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

#im_gray = cv.equalizeHist(im_gray) #CREATE ENHANCE CONTRAST FUNCTION

#im_blur = cv.blur(im, (11,11))
im_blur = cv.GaussianBlur(im_gray, (15, 15), 0)

#cv.imshow('blur', im_blur)
#cv.imshow('gaussian', im_gaussian)

# Set our filtering parameters
# Initialize parameter setting using cv.SimpleBlobDetector
params = cv.SimpleBlobDetector_Params()

# Starting, step, and ending pixel threshold
params.minThreshold = 30
params.thresholdStep = 10
params.maxThreshold = 250

# min times blob appears in threshold slices
params.minRepeatability = 2  # default=2

# min pixel length between blob sections to say its same blob
params.minDistBetweenBlobs = 10

# Set blob Color filtering parameters
params.filterByColor = True
params.blobColor = 255

# Set Area filtering parameters
params.filterByArea = True
params.minArea = 20 # 93 is 0.1% mask size of ex11.tif

# Set Circularity filtering parameters
params.filterByCircularity = True
params.minCircularity = 0.5

# Set Convexity filtering parameters
params.filterByConvexity = True
params.minConvexity = 0.8

# Set inertia filtering parameters
params.filterByInertia = False
#params.minInertiaRatio = 0.01

# Set up the detector with params
detector = cv.SimpleBlobDetector_create(params)

# Detect blobs
keypoints = detector.detect(im_blur)
#print(keypoints)

# Draw detected blobs as red circles.
# cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_keypoints = cv.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show keypoints
cv.imshow("Original", im)
#cv.imshow('Blur', im_blur)
cv.imshow("Keypoints", im_keypoints)
cv.waitKey(0)


#For blob area/circularity: get center of each blob, then perform contours in masked area so only blob is found
#Contours would find slice with largest area and then calculate area and/or circularity
#maybe use same lower param limit in contour.py, or update circularity to be above square