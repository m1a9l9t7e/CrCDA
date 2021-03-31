import argparse
import math
import os
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import cv2
import matplotlib
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans
import hdbscan
from sklearn import metrics
from tqdm import tqdm
import multiprocessing as mp
from util import stitch_images, print_red, plot
from sklearn import decomposition
from pathlib import Path
from random import shuffle
import pickle
from openTSNE import TSNE

import warnings
warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")


def calculate_features_threaded(samples):
    pool = mp.Pool(mp.cpu_count())
    features = pool.map(calculate_hog, tqdm(samples, desc='HOG multi'))
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


def reduce_feature_dims_SVD(features, n):
    """
    Reduce dimensionality of features to n
    Using truncated SVD is supposedly more memory efficient
    """
    print_red("Applying PCA (TruncatedSVD):")
    svd = decomposition.TruncatedSVD(n_components=n, algorithm='arpack')
    reduced = svd.fit_transform(features)
    return reduced


def perform_tsne(x):
    embedding = TSNE().fit(x)


def visualize_clustering(features, labels, noise=True):
    assert np.shape(features)[1] == 2

    classes = list()
    num_elements = list()
    bar_colors = list()
    # Black removed and is used for noise instead.
    unique_labels = sorted(set(labels))
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    # colors = [plt.cm.jet(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        class_member_mask = (labels == k)
        xy = features[class_member_mask]

        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=2)
            classes.append('     Noise')

        else:
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=8)
            classes.append(str(k))

        num_elements.append(int(class_member_mask.sum()))
        bar_colors.append(col)

    plt.title('t-SNE')
    plt.show()


def get_pickle_paths(input_dir, sub_dirs):
    path_lookup = dict()
    for sub_dir in sub_dirs:
        path_list = []
        path_to_sub_dir = os.path.join(input_dir, sub_dir)
        for file_name in os.listdir(path_to_sub_dir):
            path_list.append(os.path.join(path_to_sub_dir, file_name))
        path_lookup[sub_dir] = sorted(path_list)

    return path_lookup


def get_heat_map(data):
    my_cm = matplotlib.cm.get_cmap('hot')
    normed_data = (data - np.min(data)) / (np.max(data) - np.min(data))
    mapped_data = my_cm(normed_data)
    mapped_data = mapped_data[:, :, :-1]
    mapped_data = mapped_data[:, :, ::-1]
    return mapped_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Image Sampler")
    parser.add_argument("--input_dir", type=str, default='/home/malte/MA/feature_maps/GTA2Cityscapes_AdvEnt_FeatureMaps', help="Path to ground truth to extract samples from")
    parser.add_argument("--out", type=str, default="./output-t-SNE", help="Path to output folder")
    parser.add_argument("--max_samples", type=int, default=100, help="Maximum Number of Images to use")
    args = parser.parse_args()
    pca_limit = 50

    keys = ['source', 'target']
    pickle_paths = get_pickle_paths(args.input_dir, keys)
    images = dict()
    features = dict()

    for key in keys:
        feature_maps = []
        feature_vectors = []
        feature_vectors_pca = []
        for path in tqdm(pickle_paths[key][:args.max_samples], desc='Reading pickles from {}'.format(key)):
            with open(path, 'rb') as f:
                data = pickle.load(f)
            heatmaps = []
            for feature_map in data[0]:
                heatmaps.append(get_heat_map(feature_map))
            image = stitch_images(heatmaps, n_rows=32, n_cols=32)
            feature_maps.append(image)
            data = np.reshape(data, [-1])
            feature_vectors.append(data)
            if len(feature_vectors) > pca_limit:
                feature_vectors_pca.append(reduce_feature_dims(feature_vectors, 50))
                feature_vectors = []
        if len(feature_vectors) > pca_limit:
            feature_vectors_pca.append(reduce_feature_dims(feature_vectors, 50))
            feature_vectors = []
        # print('feature vectors raw: {}'.format(np.shape(feature_vectors)))
        # feature_vectors = reduce_feature_dims(feature_vectors, 50)
        # print('pca: {}'.format(np.shape(feature_vectors)))
        feature_vectors = TSNE().fit(feature_vectors_pca)
        print('t-SNE: {}'.format(np.shape(feature_vectors)))
        features[key] = feature_vectors
        images[key] = feature_maps

    feature_summary = np.concatenate([features['source'], features['target']])
    labels_summary = np.concatenate([np.zeros(len(features['source'])), np.ones(len(features['target']))])
    visualize_clustering(feature_summary, labels_summary)
    # plot(feature_summary, labels_summary)


    # for key in keys:
    #     for image in images[key]:
    #         cv2.imshow(key, image)
    #         cv2.waitKey(0)
