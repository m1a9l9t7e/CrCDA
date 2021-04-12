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
from util import stitch_images, print_red, plot, stitch_horizontal, stitch_vertical
from sklearn import decomposition
from pathlib import Path
from random import shuffle
import pickle
from openTSNE import TSNE
import scipy.sparse as sp

import warnings
warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")


def pca(x, n_components=50):
    if sp.issparse(x):
        x = x.toarray()
    U, S, V = np.linalg.svd(x, full_matrices=False)
    U[:, np.sum(V, axis=1) < 0] *= -1
    x_reduced = np.dot(U, np.diag(S))
    x_reduced = x_reduced[:, np.argsort(S)[::-1]][:, :n_components]
    return x_reduced


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
                     markeredgecolor='k', markersize=2)
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
    # parser.add_argument("--input_dir", type=str, default='/home/malte/MA/feature_maps/GTA2Cityscapes_AdvEnt_FeatureMaps', help="Path to ground truth to extract samples from")
    parser.add_argument("--input_dir", type=str, default='/home/malte/MA/feature_maps/GTA2Cityscapes_SourceOnly_FeatureMaps', help="Path to ground truth to extract samples from")
    parser.add_argument("--out", type=str, default="./output-t-SNE", help="Path to output folder")
    parser.add_argument("--tsne", type=bool, default=True, help="Path to output folder")
    parser.add_argument("--max_samples", type=int, default=500, help="Maximum Number of Images to use")
    args = parser.parse_args()
    path = Path(args.out).stem

    key = ''
    # key = 'so-'

    with open('/home/malte/PycharmProjects/Masterarbeit/CrCDA/pickles/umap-feats-SO.pickle', 'rb') as f:
        features = pickle.load(f)

    # with open('/home/malte/PycharmProjects/Masterarbeit/CrCDA/pickles/umap-feats-AD.pickle', 'rb') as f:
    #     labels_summary = pickle.load(f)

    labels_num = np.concatenate([np.zeros([int(np.shape(features)[0]/2)]), np.ones([int(np.shape(features)[0]/2)])])
    labels = ['source' if label < 0.5 else 'target' for label in labels_num]

    print(np.shape(features))
    plot(features, labels, colors={"source": "#538CBA", "target": "#8B006B"}, s=10, title='training with adaptation')
