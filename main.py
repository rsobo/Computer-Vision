# Ryan Sobolewski
# CAP 4410
# Assignment 3

import cv2
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth


def meanshift(img):
    img = cv2.medianBlur(img, 3)
    # flatten the image
    flat_image = img.reshape((-1, 3))
    flat_image = np.float32(flat_image)

    # meanshift
    bandwidth = estimate_bandwidth(flat_image, quantile=.09, n_samples=3000)
    ms = MeanShift(bandwidth=bandwidth, max_iter=800, bin_seeding=True)
    ms.fit(flat_image)
    labeled = ms.labels_

    # get the average color of each segment
    total = np.zeros((np.unique(labeled).shape[0], 3), dtype=float)
    count = np.zeros(total.shape, dtype=float)
    for i, label in enumerate(labeled):
        total[label] = total[label] + flat_image[i]
        count[label] += 1
    avg = total / count
    avg = np.uint8(avg)

    # cast the labeled image into the corresponding average color
    res = avg[labeled]
    result = res.reshape(img.shape)
    return result


def otsu(img):
    ret2, thr = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    return thr


def segmentImg(img):
    flat_image = img.reshape((-1, 3))
    flat_image = np.float32(flat_image)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(flat_image, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    result_image = res.reshape(img.shape)
    return result_image


parkbench = cv2.imread("parkBench.png")
car = cv2.imread("car.png")
man = cv2.imread("man.png", cv2.IMREAD_GRAYSCALE)
grey = cv2.imread("grey.png", cv2.IMREAD_GRAYSCALE)
wip = cv2.imread("walkInPark.png", cv2.IMREAD_GRAYSCALE)
blocks = cv2.imread("blocks.png", cv2.IMREAD_GRAYSCALE)

cv2.imshow('Mean Shift Park Bench', segmentImg(meanshift(parkbench)))
cv2.imshow('Segmented Mean Shift Car', segmentImg(meanshift(car)))

cv2.imshow('OTSU Man', segmentImg(otsu(man)))
cv2.imshow('OTSU Grey', segmentImg(otsu(grey)))

cv2.imshow('OTSU Walk in Park', segmentImg(otsu(wip)))
cv2.imshow('OTSU Blocks', segmentImg(otsu(blocks)))

cv2.waitKey(0)
cv2.destroyAllWindows()
