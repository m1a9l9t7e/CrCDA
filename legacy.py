import cv2
from skimage.exposure import exposure
from skimage.feature import hog
import matplotlib.pyplot as plt
import numpy as np


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