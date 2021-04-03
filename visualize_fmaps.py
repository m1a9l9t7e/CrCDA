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
# from sklearn.manifold import TSNE
import scipy.sparse as sp
import umap
import sys
import warnings
warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")

maps_per_layer = {'layer1': 256,
                  'layer2': 512,
                  'layer3': 1024,
                  'layer4': 2048}

def calculate_features_threaded(samples):
    pool = mp.Pool(mp.cpu_count())
    features = pool.map(calculate_hog, tqdm(samples, desc='HOG multi'))
    pool.close()
    return features


def pca(x, n_components=50):
    if sp.issparse(x):
        x = x.toarray()
    U, S, V = np.linalg.svd(x, full_matrices=False)
    U[:, np.sum(V, axis=1) < 0] *= -1
    x_reduced = np.dot(U, np.diag(S))
    x_reduced = x_reduced[:, np.argsort(S)[::-1]][:, :n_components]
    return x_reduced


def get_pickle_paths(input_dir, sub_dirs):
    path_lookup = dict()
    for sub_dir in sub_dirs:
        path_list = []
        path_to_sub_dir = os.path.join(input_dir, sub_dir)
        for file_name in os.listdir(path_to_sub_dir):
            path_list.append(os.path.join(path_to_sub_dir, file_name))
        path_lookup[sub_dir] = sorted(path_list)

    return path_lookup


def get_heat_map(data, color='hot'):
    my_cm = matplotlib.cm.get_cmap(color)
    normed_data = (data - np.min(data)) / (np.max(data) - np.min(data))
    mapped_data = my_cm(normed_data)
    mapped_data = mapped_data[:, :, :-1]
    mapped_data = mapped_data[:, :, ::-1]
    return mapped_data


def assemble_heat_maps(data, num_fmaps, resize=(63, 35), color='hot'):
    heatmaps = []
    for index, feature_map in enumerate(data[0]):
        if resize is not None:
            feature_map = cv2.resize(feature_map, resize)
        heatmap = get_heat_map(feature_map, color=color)
        heatmaps.append((heatmap * 255).astype(np.uint8))
    image = stitch_images(heatmaps, n_rows=int(num_fmaps ** 0.5), n_cols=int(num_fmaps ** 0.5), padding=2)
    return image


def resize_feature_maps(fmaps, resize=(63, 35)):
    resized = []
    for index, feature_map in enumerate(fmaps[0]):
        feature_map = cv2.resize(feature_map, resize)
        resized.append(feature_map)
    return np.expand_dims(np.array(resized), axis=0)


def get_info_from_input_dir_name(input_dir_name, seperator='_'):
    num_fmaps = None
    split = input_dir_name.split(seperator)
    for item in split:
        if 'layer' in item:
            num_fmaps = maps_per_layer[item]

    return num_fmaps


def get_output_path(output_dir, input_dir_name, viz, pca_n):
    stem = '{}-{}-{}'.format(viz, pca_n, input_dir_name)
    out_path = os.path.join(output_dir, stem)
    os.makedirs(out_path, exist_ok=True)
    return out_path


def get_size(pickle_paths):
    size = [-1, -1]
    for key in pickle_paths.keys():
        first_path = pickle_paths[key][0]
        with open(first_path, 'rb') as f:
            data = pickle.load(f)
            first_fmap = np.squeeze(data)[0]
            _size = np.shape(np.squeeze(first_fmap))
            for i in [0, 1]:
                if _size[i] > size[i]:
                    size[i] = _size[i]

    size = (size[1], size[0])
    return size


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Image Sampler")
    parser.add_argument("--input_dir", type=str, default='/home/malte/MA/feature_maps/GTA2Cityscapes_AdvEnt_FeatureMaps_layer1', help="Path to ground truth to extract samples from")
    parser.add_argument("--out", type=str, default="./viz-output", help="Path to output folder")
    # parser.add_argument("--viz", type=str, default='umap', choices=['tsne', 'umap'], help="type of low dim viz")
    parser.add_argument("--viz", type=str, default=None, choices=['tsne', 'umap'], help="type of low dim viz")
    parser.add_argument("--fmaps", type=str, default='mean', choices=['single', 'mean'], help="show feature map visualization")
    parser.add_argument("--pca_n", type=int, default=4096, help="Principal components to use for dim reduction")
    parser.add_argument("--uniform_size", type=bool, default=True, help="resize source and target feature maps to be the same size")
    parser.add_argument("--max_samples", type=int, default=500, help="Maximum Number of Images to use")
    args = parser.parse_args()

    pca_limit = 50
    output_path = get_output_path(args.out, Path(args.input_dir).stem, args.viz, args.pca_n)
    print_red(output_path)
    reducer = umap.UMAP()
    keys = ['source', 'target']
    pickle_paths = get_pickle_paths(args.input_dir, keys)
    size = None
    if args.uniform_size:
        size = get_size(pickle_paths)
    num_fmaps = get_info_from_input_dir_name(Path(args.input_dir).stem)

    feature_maps = dict()
    heat_maps = dict()
    features = dict()

    for key in keys:
        _feature_maps = []
        _heat_maps = []
        feature_vectors = []
        feature_vectors_pca = []
        iterator = tqdm(pickle_paths[key][:args.max_samples], desc='Reading pickles from {}'.format(key))
        for path in iterator:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            if args.fmaps:
                _feature_maps.append(data[0])
                _heat_maps.append(assemble_heat_maps(data, num_fmaps, resize=size))
            if args.viz:
                data = np.reshape(data, [-1])
                feature_vectors.append(data)
                if len(feature_vectors) >= pca_limit:
                    iterator.set_description("Applying pca to batch of {} samples".format(len(feature_vectors)))
                    for feature_vector in pca(feature_vectors, args.pca_n):
                        feature_vectors_pca.append(feature_vector)
                    feature_vectors = []
        if len(feature_vectors) > 0:
            print_red("\npca_limit does not divide max_samples evenly. {} samples were not processed!".format(len(feature_vectors)))

        feature_vectors_pca = np.array(feature_vectors_pca)
        print('features after PCA: {}'.format(np.shape(feature_vectors_pca)))
        if args.viz == 'tsne':
            feature_vectors = TSNE().fit(feature_vectors_pca)  # openTSNE implementation
            # feature_vectors = TSNE(n_components=2).fit_transform(feature_vectors_pca)  # sklearn implementation
            print('Applying t-SNE: {}'.format(np.shape(feature_vectors)))
        elif args.viz == 'umap':
            feature_vectors = reducer.fit_transform(feature_vectors_pca)
            print('Applying UMAP: {}'.format(np.shape(feature_vectors)))

            features[key] = feature_vectors
        heat_maps[key] = _heat_maps
        feature_maps[key] = _feature_maps

    if args.viz:
        feature_summary = np.concatenate([features['source'], features['target']])
        labels_summary = np.concatenate([np.zeros(len(features['source'])), np.ones(len(features['target']))])
        labels_summary_text = ['source' if label < 0.5 else 'target' for label in labels_summary]
        plot(feature_summary, labels_summary_text, colors={"source": "#538CBA", "target": "#8B006B"}, s=10, save_path=os.path.join(output_path, 'plot.png'))
        with open(os.path.join(output_path, 'feats.pickle'), "wb") as f_out:
            pickle.dump(feature_summary, f_out)
        with open(os.path.join('label.pickle'), "wb") as f_out:
            pickle.dump(labels_summary, f_out)

    # mean heat maps
    if args.fmaps == 'mean':
        mean_list = []
        for key in keys:
            print('Shape of all feature maps for key {}: {}'.format(key, np.shape(feature_maps[key])))
            mean_feature_maps = np.mean(feature_maps[key], axis=0)
            print('Shape of mean feature map: {}'.format(np.shape(mean_feature_maps)))
            mean_feature_maps = np.expand_dims(mean_feature_maps, axis=0)
            mean_list.append(resize_feature_maps(mean_feature_maps, size))
            mean_heat_map = assemble_heat_maps(mean_feature_maps, num_fmaps, resize=size)
            cv2.imshow('win', mean_heat_map)
            cv2.waitKey(0)
            cv2.imwrite(os.path.join(output_path, key + '-mean-heatmap.png'), mean_heat_map, [cv2.IMWRITE_PNG_COMPRESSION, 0])

        difference = np.abs(np.subtract(mean_list[0], mean_list[1]))
        difference_heat_map = assemble_heat_maps(difference, num_fmaps, resize=None, color='hot')
        cv2.imshow('win', difference_heat_map)
        cv2.waitKey(0)
        cv2.imwrite(os.path.join(output_path, 'difference-mean-heatmap.png'), difference_heat_map, [cv2.IMWRITE_PNG_COMPRESSION, 0])

        # colors = ['binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
        #           'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
        #           'hot', 'afmhot', 'gist_heat', 'copper']
        #
        # for color in colors:
        #     difference = np.abs(np.subtract(mean_list[0], mean_list[1]))
        #     difference_heat_map = assemble_heat_maps(difference, num_fmaps, resize=None, color=color)
        #     cv2.imshow('win', difference_heat_map)
        #     cv2.waitKey(0)

    # single heat maps
    if args.fmaps == 'single':
        for key in keys:
            for image in heat_maps[key][:1]:
                image = image.astype(np.uint8)
                cv2.imshow(key, image)
                cv2.waitKey(0)
                cv2.imwrite(os.path.join(output_path, key + '-heatmap.png'), image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
