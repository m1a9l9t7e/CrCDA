import argparse
import os

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn import metrics
from tqdm import tqdm
import multiprocessing as mp


def print_red(string):
    print('\u001B[31m' + string + '\u001B[0m')


def extract_samples(image_path):
    """
    Extracts samples from image and adds them to self.samples
    :param image_path: path to image
    :param sample_shape: shape of sample (H x W x C)
    :return: array of samples
    """
    image = cv2.imread(image_path)
    samples = sliding_window_view(image, sample_shape)[0::sample_shape[0], 0::sample_shape[1]]
    samples = samples.reshape([-1] + sample_shape)
    return samples


def calculate_features_threaded(samples):
    pool = mp.Pool(mp.cpu_count())
    features = pool.map(hog, tqdm(samples, desc='HOG multi'))
    pool.close()
    return features


def reduce_feature_dims(features, n):
    """
    Reduce dimensionality of features to n
    """
    print_red("Applying PCA:")
    pca = PCA(n_components=n)
    reduced = pca.fit_transform(features)
    return reduced


def cluster_samples(features, eps=0.1, min_samples=10):
    """
    Clusters samples in self.samples by their corresponding feature vector.
    """
    # features = StandardScaler().fit_transform(features)  # needed?
    if np.shape(features)[1] > 2:
        features_2d = reduce_feature_dims(features, 2)
    else:
        features_2d = features

    print_red("Finding Clusters with DBSCAN:")
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(features)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(features, labels))

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = features_2d[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)

        xy = features_2d[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()

    return labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Image Sampler")
    parser.add_argument("--label_dir", type=str, default='./labels', help="Path to ground truth to extract samples from")
    parser.add_argument("--out", type=str, default="./samples", help="Path to output folder")
    args = parser.parse_args()

    img = cv2.imread('./labels/00001.png')
    sample_shape = [64, 64, img.shape[2]]

    # 1. Extract Samples from Ground Truth Images
    image_paths = [os.path.join('./labels', image_name) for image_name in os.listdir('./labels')[:100]]
    pool = mp.Pool(mp.cpu_count())
    result = pool.map(extract_samples, tqdm(image_paths, desc='Extract Samples'))
    samples = np.reshape(result, [-1] + sample_shape)
    pool.close()
    print("Extracted Raw Samples: {}".format(np.shape(samples)))

    # 2. Calculate HOG Feature vector for each sample
    feature_vectors = calculate_features_threaded(samples)
    print("HOG Features Vectors: {}".format(np.shape(feature_vectors)))

    # 3. Reduce Dimensionality of HOG Feature vectors to 2
    feature_vectors_2d = reduce_feature_dims(feature_vectors, 2)
    print("PCA Feature Vectors: {}".format(np.shape(feature_vectors_2d)))

    # 4. Cluster Samples according to their 2d HOG features
    labels = cluster_samples(feature_vectors_2d)
    print("Sample Cluster Labels: {}".format(np.shape(labels)))
