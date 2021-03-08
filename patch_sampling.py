import argparse
import os
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import hdbscan
from sklearn import metrics
from tqdm import tqdm
import multiprocessing as mp
from util import stitch_images, print_red
from sklearn import decomposition
from pathlib import Path


def extract_samples(image_path):
    """
    Extracts samples with shape sample_shape from single image with equal stride
    :param image_path: path to image
    :param sample_shape: shape of sample (H x W x C)
    :return: array of samples
    """
    image = cv2.imread(image_path)
    image = cv2.resize(image, (1280, 720))
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


def reduce_feature_dims_SVD(features, n):
    """
    Reduce dimensionality of features to n
    Using truncated SVD is supposedly more memory efficient
    """
    print_red("Applying PCA (TruncatedSVD):")
    svd = decomposition.TruncatedSVD(n_components=n, algorithm='arpack')
    reduced = svd.fit_transform(features)
    return reduced


def cluster_samples(features, eps=0.1, min_samples=10, visualize=False):
    """
    Clusters feature vectors using DBSCAN and show result of 2d transformation
    """
    # features = StandardScaler().fit_transform(features)  # needed?

    print_red("Finding clusters with DBSCAN:")
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(features)
    # db = DBSCAN(eps=eps, min_samples=min_samples, algorithm='ball_tree').fit(features)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(features, labels))

    if visualize:
        visualize_clustering(features, labels)
        if np.shape(features)[1] > 2:
            features_2d = reduce_feature_dims(features, 2)
        else:
            features_2d = features

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


def cluster_samples_HDBSCAN(features, min_samples=10, visualize=False):
    """
    Clusters feature vectors using DBSCAN and show result of 2d transformation
    """
    # features = StandardScaler().fit_transform(features)  # needed?

    print_red("Finding clusters with HDBSCAN:")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_samples)
    labels = clusterer.fit_predict(features)

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)

    if visualize:
        visualize_clustering(features, labels)

    return labels


def visualize_clustering(features, labels):
    if np.shape(features)[1] > 2:
        features_2d = reduce_feature_dims(features, 2)
    else:
        features_2d = features

    classes = list()
    num_elements = list()
    bar_colors = list()
    # Black removed and is used for noise instead.
    unique_labels = sorted(set(labels))
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    # colors = [plt.cm.jet(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        class_member_mask = (labels == k)
        xy = features_2d[class_member_mask]

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

    plt.title('Number of clusters: %d' % len(unique_labels))
    plt.show()

    # Move Noise Class to last
    classes.append(classes.pop(0))
    num_elements.append(num_elements.pop(0))
    bar_colors.append(bar_colors.pop(0))
    plt.title('Cluster distribution')
    plt.bar(classes, num_elements, color=bar_colors)
    plt.show()


def show_cluster_examples(samples, labels, n=64):
    cluster_ids = set(labels)
    cluster_id_to_samples_map = dict()
    for cluster_id in cluster_ids:
        cluster_id_to_samples_map[cluster_id] = list()

    for idx, label in enumerate(labels):
        if len(cluster_id_to_samples_map[label]) < n:
            cluster_id_to_samples_map[label].append(samples[idx])

    for cluster_id in cluster_ids:
        _cluster_samples = cluster_id_to_samples_map[cluster_id]
        image = stitch_images(_cluster_samples, n_rows=8, n_cols=8)
        cv2.imshow("c-id", image)
        cv2.setWindowTitle("c-id", 'Cluster {} samples (n={})'.format(cluster_id if cluster_id >= 0 else "NOISE", list(labels).count(cluster_id)))
        cv2.waitKey(0)


def get_image_labels(samples, labels, image_shape, sample_shape, one_hot_encoding=True, show_segmented=False):
    print_red("Generating image labels{}:".format(" with one hot encoding" if one_hot_encoding else ""))
    w = int(image_shape[1] / sample_shape[1])
    h = int(image_shape[0] / sample_shape[0])

    if show_segmented:
        segmented_img = stitch_images(list(samples[:h*w]), h, w)
        cv2.imshow('win', segmented_img)
        cv2.waitKey(0)

    labels = labels + 1  # Needed as currently there is a -1 class for Noise

    if one_hot_encoding:
        n_classes = len(set(labels))
        one_hot_targets = np.eye(n_classes)[labels]
        return np.reshape(one_hot_targets, [-1, h, w, n_classes])
    else:
        return np.reshape(labels, [-1, h, w])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Image Sampler")
    parser.add_argument("--label_dir", type=str, default='./data/gta5_labels', help="Path to ground truth to extract samples from")
    parser.add_argument("--out", type=str, default="./output", help="Path to output folder")
    parser.add_argument("--max_images", type=int, default=500, help="Maximum Number of Images to use")
    args = parser.parse_args()

    img = cv2.resize(cv2.imread(os.path.join(args.label_dir, os.listdir(args.label_dir)[0])), (1280, 720))
    # sample_shape = [18, 32, img.shape[2]]  # first size
    sample_shape = [36, 64, img.shape[2]]  # second size

    # 1. Extract Samples from Ground Truth Images
    image_paths = sorted([os.path.join(args.label_dir, image_name) for image_name in os.listdir(args.label_dir)])[:args.max_images]
    pool = mp.Pool(mp.cpu_count())
    result = pool.map(extract_samples, tqdm(image_paths, desc='Extract samples'))
    samples = np.reshape(result, [-1] + sample_shape)
    pool.close()
    print("Extracted Raw Samples: {}".format(np.shape(samples)))

    # 2. Calculate HOG Feature vector for each sample
    feature_vectors = calculate_features_threaded(samples)
    print("HOG Feature Vectors: {}".format(np.shape(feature_vectors)))

    # 3. Reduce Dimensionality of HOG Feature vectors to n
    feature_vectors_nd = reduce_feature_dims(feature_vectors, n=12)
    print("PCA Feature Vectors: {}".format(np.shape(feature_vectors_nd)))

    # 4. Cluster Samples according to their nd HOG features
    # labels = cluster_samples(feature_vectors_nd, eps=0.1, min_samples=10, visualize=True)
    labels = cluster_samples_HDBSCAN(feature_vectors_nd, min_samples=500)
    print("Sample Cluster Labels: {}".format(np.shape(labels)))

    # 5. Show Example Samples for each Cluster
    show_cluster_examples(samples, labels, n=64)

    # 6. Generate 2d image-level labels for each image
    image_labels = get_image_labels(samples, labels, np.shape(img), sample_shape, one_hot_encoding=False, show_segmented=False)
    print("Image Labels: {}".format(np.shape(image_labels)))

    # 7. Save labels as pngs
    os.makedirs(args.out, exist_ok=True)
    for idx, image_label in enumerate(image_labels):
        cv2.imwrite(os.path.join(args.out, Path(image_paths[idx]).stem + '.png'), image_label)

    # 8. Since the Original Cluster can't contain all samples, we need to assign the remaining samples a label after the fact.
    # This Problem is discussed here: https://stackoverflow.com/questions/27822752/scikit-learn-predicting-new-points-with-dbscan
