import cv2
import numpy as np

# Read image
im = cv2.imread("ex11.tif", cv2.IMREAD_GRAYSCALE)
# im = cv2.equalizeHist(im) #CREATE ENHANCE CONTRAST FUNCTION
im_blur = cv2.blur(im, (11,11))

# Set our filtering parameters
# Initialize parameter setting using cv2.SimpleBlobDetector
params = cv2.SimpleBlobDetector_Params()

# Starting, step, and ending pixel threshold
params.minThreshold = 100
params.thresholdStep = 10
params.maxThreshold = 250

# min times blob appears in threshold slices
params.minRepeatability = 3  # default=2

# min pixel length between blobs
params.minDistBetweenBlobs = 5

# Set blob Color filtering parameters
params.filterByColor = True
params.blobColor = 255

# Set Area filtering parameters
params.filterByArea = False
#params.minArea = 100
 
# Set Circularity filtering parameters
params.filterByCircularity = False
#params.minCircularity = 0.9
 
# Set Convexity filtering parameters
params.filterByConvexity = False
#params.minConvexity = 0.2
     
# Set inertia filtering parameters
params.filterByInertia = False
#params.minInertiaRatio = 0.01

# Set up the detector with params
detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs
keypoints = detector.detect(im_blur)
#print(keypoints)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show keypoints
cv2.imshow("Original", im)
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)