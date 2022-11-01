import numpy as np
import cv2
from matplotlib import pyplot as plt

# img = cv2.imread('coins.png')
img = cv2.imread('sic_pre_1-slice2-adjusted.tif')
# img = cv2.imread('ex11.tif')


gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
im_blur = cv2.GaussianBlur(gray,(15,15),0)
ret, thresh = cv2.threshold(im_blur,135,255,1)
new_thresh = cv2.bitwise_not(thresh)

# noise removal
kernel = np.ones((2,2),np.uint8)
opening = cv2.morphologyEx(new_thresh,cv2.MORPH_OPEN,kernel, iterations = 1)


# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=8)
# cv2.imshow("sure_bg", sure_bg)


# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,3)
ret, sure_fg = cv2.threshold(dist_transform,0.35*dist_transform.max(),255,0)
cv2.imshow("opening", opening)


# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)
cv2.imshow("unknown", unknown)


# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)


# Add one to all labels so that sure background is not 0, but 1
markers = markers+1

# Now, mark the region of unknown with zero
markers[unknown==255] = 0



markers = cv2.watershed(img,markers)
img[markers == -1] = [255,0,0]


cv2.imshow("Test", img)
cv2.waitKey(0)
