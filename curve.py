import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def gradient_spaced(L,num):
    grad = np.array([(L[i+num] - L[i-num])/(num*2) for i in range(-num,len(L)-num)])
    # reorder matrix to correct indices
    grad = np.append(grad[num:], grad[0:num])
    return grad

def curvature(contour, num):
    
    dx_dt = gradient_spaced(contour[:, 0], num)
    dy_dt = gradient_spaced(contour[:, 1], num)

    # velocity
    vel = np.array([[dx_dt[i], dy_dt[i]] for i in range(dx_dt.size)])

    # speed
    ds_dt = np.sqrt(dx_dt * dx_dt + dy_dt * dy_dt)

    # unit tangent vector
    tangent = np.array([1/ds_dt] * 2).transpose() * vel

    d2s_dt2 = gradient_spaced(ds_dt, num)
    d2x_dt2 = gradient_spaced(dx_dt, num)
    d2y_dt2 = gradient_spaced(dy_dt, num)

    curvature = np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt * dx_dt + dy_dt * dy_dt)**1.5    
    
    return curvature
    
im = cv.imread("ex3.tif")
im_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
im_blur = cv.GaussianBlur(im_gray, (25, 25), 0)
ret, im_thresh = cv.threshold(im_blur, 25, 255, cv.THRESH_BINARY)
contours, hierarchy = cv.findContours(im_thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
contour = max(contours, key=cv.contourArea)
contour = contour[:,0]

im2 = cv.imread("ex3.tif")
im_gray2 = cv.cvtColor(im2, cv.COLOR_BGR2GRAY)
im_blur2 = cv.GaussianBlur(im_gray2, (25, 25), 0)
ret2, im_thresh2 = cv.threshold(im_blur2, 65, 255, cv.THRESH_BINARY)
contours2, hierarchy2 = cv.findContours(im_thresh2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
contour2 = max(contours2, key=cv.contourArea)
contour2 = contour2[:,0]


curv1 = curvature(contour,5)
curv2 = curvature(contour2,5)
curv = np.append(curv1,curv2)

print('mean1: ' + str(np.mean(curv1)))
print('mean2: ' + str(np.mean(curv2)))
print('std1: ' + str(np.std(curv1)))
print('std2: ' + str(np.std(curv2)))

unique_curv = np.unique(curv)
colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_curv)))

fig, (ax1, ax2) = plt.subplots(1, 2)

for i in range(len(curv1)):
    idx1 = np.where(unique_curv == curv1[i])
    ax1.plot(contour[i][0], contour[i][1], '.', color=colors[idx1])
    ax1.title.set_text('curv1')

for i in range(len(curv2)):
    idx2 = np.where(unique_curv == curv2[i])
    ax2.plot(contour2[i][0], contour2[i][1], '.', color=colors[idx2])
    ax2.title.set_text('curv2')

plt.show()

