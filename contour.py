'''
This code uses the active_contours function from scikit-image to create a snake that fits to the nucleus of the cell
'''
from skimage.draw import circle_perimeter
from skimage.filters import gaussian, sobel
from skimage.color import rgb2gray
from skimage import io
from skimage.segmentation import active_contour
from skimage.feature import canny
import numpy as np
import matplotlib.pyplot as plt
import contrast_function_scikit as cfs

input_name = 'test_2.jpg'
output_name_contrast = 'contrasted_image.png'
output_name = 'new_test_2.png'

def contour(input_name, output_name_contrast, output_name):
    
    contrast_image = cfs.contrast(input_name, output_name_contrast)
    ## Active_Contour
    img = io.imread(contrast_image)
    img = rgb2gray(img)

    s = np.linspace(0, 2*np.pi, 1000)
    r = 270 + 280*np.sin(s)
    c = 260 + 270*np.cos(s)
    init = np.array([r, c]).T

    snake = active_contour(sobel(img),
                        init, w_line=0, w_edge=1, alpha=0.095, gamma=0.001)

#print(snake)

# contour_perimeter = cv.arcLength(snake, True)
# contour_circularity = bp.get_circularity(snake)

# print('Contour Area: ' + str(contour_area))
# print('Contour Perimeter: ' + str(round(contour_perimeter,1)))
# print('Contour Circularity: ' + str(round(contour_circularity,3)))

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img, cmap=plt.cm.gray)
    ax.plot(init[:, 1], init[:, 0], '--r', lw=3)
    ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
    ax.set_xticks([]), ax.set_yticks([])
    ax.axis([0, img.shape[1], img.shape[0], 0])

    plot = plt.savefig(output_name, bbox_inches='tight')
    return output_name

contour(input_name, output_name_contrast, output_name)




''' This code takes mulitple binary images at multiple thresholds
    and then finds the contours.
    The contour with the highest total area is selected to be drawn.
    Will have to add a min_threshold parameter '''

# import cv2 as cv
# import numpy as np
# import blob_params as bp

# # Load source image
# im = cv.imread("ex3.tif")
# if im is None:
#     print('Could not open or find the image:', args.input)
#     exit(0)

# # Gray and Blur image
# im_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
# #im_gray = cv.equalizeHist(im_gray)
# im_blur = cv.GaussianBlur(im_gray, (25,25), 0)

# contour_list = []
# mask_intensity_list = []
# for i in range(25,255,10):  # add min threshold for image parameter
#     ret, im_thresh = cv.threshold(im_blur, i, 255, cv.THRESH_BINARY)
#     # cv.imshow('binary'+str(i), im_thresh)
#     # Find only the most external contours
#     contours, hierarchy = cv.findContours(im_thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
#     if len(contours) > 0:
#         for j in range(len(contours)):
#             contour_list.append(contours[j])  # add contours to main list
#             mask_intensity_list.append(i)

# # find the contour with the highest area
# max_contour = max(contour_list, key=cv.contourArea)
# max_contour_idx = contour_list.index(max_contour)
# mask_intensity = mask_intensity_list[max_contour_idx]
# ret, mask = cv.threshold(im_blur, mask_intensity, 255, cv.THRESH_BINARY)
# im_mask = cv.bitwise_and(im, im, mask = mask)

# contour_area = cv.contourArea(max_contour)
# contour_perimeter = cv.arcLength(max_contour, True)
# contour_circularity = bp.get_circularity(max_contour)
# cx, cy = bp.get_center(max_contour)

# print('Contour Area: ' + str(contour_area))
# print('Contour Perimeter: ' + str(round(contour_perimeter,1)))
# print('Contour Circularity: ' + str(round(contour_circularity,3)))
# print('Contour Center: (' + str(cx) + ',' + str(cy) + ')')

# # draw the max contour
# im_copy = im.copy()
# cv.drawContours(im_copy, max_contour, -1, (0, 255, 0), 2, cv.LINE_8)
# cv.circle(im_copy, (cx,cy), 5, (0,0,255), -1)

# # Display image
# cv.imshow('source_window', im)
# #cv.imshow('blur', im_blur)
# #cv.imshow('binary', im_thresh)
# cv.imshow('Contours', im_copy)
# cv.imshow('Masked', im_mask)

# cv.waitKey()


'''
# This code is for a single threshold; draws all contours; for reference

# Load source image
im = cv.imread("ex3.tif")
if im is None:
    print('Could not open or find the image:', args.input)
    exit(0)
    
im_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
im_blur = cv.GaussianBlur(im_gray, (25,25), 0)

ret, im_thresh = cv.threshold(im_blur, 127, 255, cv.THRESH_BINARY)

# Find contours
contours, hierarchy = cv.findContours(im_thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
print(contours)

# Draw contours
im_copy = im.copy()
cv.drawContours(im_copy, contours, -1, (0, 255, 0), 2, cv.LINE_8)

# Show in a window
cv.imshow('source_window', im)
cv.imshow('blur', im_blur)
cv.imshow('binary', im_thresh)
cv.imshow('Contours', im_copy)

cv.waitKey()
'''


