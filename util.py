import cv2
from skimage.exposure import exposure
from skimage.feature import hog
import matplotlib.pyplot as plt
import numpy as np


def print_red(string):
    print('\u001B[31m' + string + '\u001B[0m')


def stitch_horizontal(image_top, image_bottom, padding):
    _shape = np.shape(image_top)
    image = np.concatenate([image_top, np.ones([padding, _shape[1], _shape[2]], dtype=np.uint8) * 255, image_bottom], axis=0)
    return image


def stitch_vertical(image_left, image_right, padding):
    _shape = np.shape(image_left)
    image = np.concatenate([image_left, np.ones([_shape[0], padding, _shape[2]], dtype=np.uint8) * 255, image_right], axis=1)
    return image


def stitch_images(images, n_rows, n_cols, padding=1):
    if len(images) < n_rows * n_cols:
        for i in range(n_rows * n_cols - len(images)):
            images.append(np.ones(np.shape(images[0]), dtype=np.uint8) * 255)

    rows = []
    for _ in range(n_rows):
        row = images.pop(0)
        for _ in range(n_cols - 1):
            row = stitch_vertical(image_left=row, image_right=images.pop(0), padding=padding)
        rows.append(row)

    image = rows.pop(0)
    for row in rows:
        image = stitch_horizontal(image_top=image, image_bottom=row, padding=padding)

    return image


def get_hog_scikit(image):
    """
    Visualization of scikit HOG features
    :param image:
    :return:
    """
    feature_vector, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True, multichannel=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    ax1.axis('off')
    ax1.imshow(image, cmap=plt.cm.gray)
    ax1.set_title('Input image')

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    plt.show()
    print("scikit feature vector: {}".format(np.shape(feature_vector)))
    return feature_vector


def get_hog_cv2(image):
    """
    Alternative way of extracting HOG features
    :param image:
    :return:
    """
    winSize = (32, 32)
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    nbins = 9
    derivAperture = 1
    winSigma = 4.
    histogramNormType = 0
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = 0
    nlevels = 64
    hog_descriptor = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                                       histogramNormType, L2HysThreshold, gammaCorrection, nlevels)

    feature_vector = hog_descriptor.compute(image)
    # print("cv2 feature vector: {}".format(np.shape(feature_vector)))
    return feature_vector


if __name__ == '__main__':
    image = stitch_images([cv2.resize(cv2.imread('./labels/00001.png'), (160, 90)) for i in range(16)], n_rows=4, n_cols=3, padding=1)
    cv2.imshow('win', image)
    cv2.waitKey(0)
