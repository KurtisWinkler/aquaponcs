import numpy as np
import cv2
import sys

def main():
    cv2.imshow("img_result", imageSegmentation('siL_pre_6-slice2-adjusted.tif'))
    cv2.waitKey(0)


def imageSegmentation(filename):
    """
    Use marker-based image segmentation using watershed algorithm.
    Label the region which being the foreground or object with one intensity, label the region
    which being background or non-object with another intensity and finally the region which are
    not sure of anything, label it with 0. That is marker. Then apply watershed algorithm. Then the
    marker will be updated with the labels gaven, and the boundaries of objects will have a value
    of -1.
    --------------------------------
    :param filename: The original image to be processed.
    :return: The image has segmented chromatins.
    """

    try:
        img = cv2.imread(filename)
    except:
        FileNotFoundError
        sys.exit(1)

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    im_blur = cv2.GaussianBlur(gray,(15,15),0)
    ret, thresh = cv2.threshold(im_blur,130,255,1)
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
    # cv2.imshow("opening", opening)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    # cv2.imshow("unknown", unknown)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1

    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0

    markers = cv2.watershed(img,markers)
    img[markers == -1] = [255,0,0]
    return img


if __name__ == '__main__':
    main()
