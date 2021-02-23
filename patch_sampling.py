import argparse
import os

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm



class Sampler:

    samples = None
    features = None

    def __init__(self, sample_shape):  # sample_shape has form (H x W x C)
        self.sample_shape = sample_shape

    def extract_samples(self, image):
        """
        Extracts samples from image and adds them to self.samples
        """
        samples = sliding_window_view(image, self.sample_shape)[0::self.sample_shape[0], 0::self.sample_shape[1]]
        samples = samples.reshape([-1] + self.sample_shape)
        if self.samples is None:
            self.samples = samples
        else:
            self.samples = np.concatenate([self.samples, samples], axis=0)

        # print("Extracted Raw Samples:\t{}".format(np.shape(self.samples)))
        return samples

    def calculate_features(self):
        """
        Calculates features for every sample in self.samples
        """
        self.features = None
        for sample in tqdm(self.samples, desc='Calculating HOG features'):
            features = get_hog_scikit(sample)
            if self.features is None:
                self.features = [features]
            else:
                self.features = np.concatenate((self.features, [features]))

        print("HOG Feature Vectors:\t{}".format(np.shape(self.features)))
        return self.features

    def reduce_feature_dims(self, n):
        """
        Reduce dimensionality of features to n
        """
        print("Applying PCA...")
        pca = PCA(n_components=n)
        reduced = pca.fit_transform(self.features)
        print("PCA Feature Vectors:\t{}".format(np.shape(reduced)))
        return reduced

    def cluster_samples(self, X):
        """
        Clusters samples in self.samples by their corresponding feature vector.
        """
        X = StandardScaler().fit_transform(X)

        print("Finding Clusters with DBSCAN...")
        db = DBSCAN(eps=0.3, min_samples=10).fit(X)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        print('Estimated number of clusters: %d' % n_clusters_)
        print('Estimated number of noise points: %d' % n_noise_)
        print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))

        # Black removed and is used for noise instead.
        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each)
                  for each in np.linspace(0, 1, len(unique_labels))]
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = (labels == k)

            xy = X[class_member_mask & core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=14)

            xy = X[class_member_mask & ~core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=6)

        plt.title('Estimated number of clusters: %d' % n_clusters_)
        plt.show()


def get_hog_scikit(image):
    feature_vector, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True, multichannel=True)

    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
    #
    # ax1.axis('off')
    # ax1.imshow(image, cmap=plt.cm.gray)
    # ax1.set_title('Input image')
    #
    # # Rescale histogram for better display
    # hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    #
    # ax2.axis('off')
    # ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    # ax2.set_title('Histogram of Oriented Gradients')
    # plt.show()
    # print("scikit feature vector: {}".format(np.shape(feature_vector)))
    return feature_vector


def get_hog_cv2(image):
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
    parser = argparse.ArgumentParser(description="Image Sampler")
    parser.add_argument("--label_dir", type=str, default='./labels', help="Path to ground truth to extract samples from")
    parser.add_argument("--out", type=str, default="./samples", help="Path to output folder")
    args = parser.parse_args()

    # # numbers test
    # img = np.linspace((0, 10, 20, 30, 40, 50), (10, 20, 30, 40, 50, 60), 11)
    # sample_extractor = Sampler(sample_shape=[3, 3])
    # print(sample_extractor.extract_samples(img))

    sample_extractor = None
    for idx, image_name in enumerate(tqdm(os.listdir('./labels'), desc='Extracting Raw Samples')):
        image_path = os.path.join('./labels', image_name)
        img = cv2.imread(image_path)

        if sample_extractor is None:
            sample_extractor = Sampler(sample_shape=[64, 64, img.shape[2]])

        sample_extractor.extract_samples(img)

        if idx > 100:
            break

    print("Extracted Raw Samples:\t{}".format(np.shape(sample_extractor.samples)))

    feature_vectors = sample_extractor.calculate_features()
    feature_vectors_2d = sample_extractor.reduce_feature_dims(2)
    sample_extractor.cluster_samples(feature_vectors_2d)
    # print(X)
