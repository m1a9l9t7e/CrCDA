import argparse
import os
import numpy as np
import cv2
import matplotlib
from util import stitch_images, plot, stitch_horizontal, stitch_vertical
from sklearn import decomposition
from pathlib import Path
import pickle
from openTSNE import TSNE
# from sklearn.manifold import TSNE
import scipy.sparse as sp
import sys
import timeit
import warnings

warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")

heat_map_colors = ['binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
                   'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
                   'hot', 'afmhot', 'gist_heat', 'copper']

maps_per_layer = {'layer1': 256,
                  'layer2': 512,
                  'layer3': 1024,
                  'layer4': 2048}

color_dict = {"source": "#BE1E3C",
              "target": "#FA6E00",  # "#FFCD00",
              "source_adapt": "#0080B4",
              "target_adapt":  "#89A400",
              }


def pca(x, n_components):
    if sp.issparse(x):
        x = x.toarray()
    U, S, V = np.linalg.svd(x, full_matrices=False)
    U[:, np.sum(V, axis=1) < 0] *= -1
    x_reduced = np.dot(U, np.diag(S))
    x_reduced = x_reduced[:, np.argsort(S)[::-1]][:, :n_components]
    return x_reduced


def reduce_feature_dims_SVD(features, n):
    """
    Reduce dimensionality of features to n
    Using truncated SVD is supposedly more memory efficient
    """
    svd = decomposition.TruncatedSVD(n_components=n, algorithm='arpack')
    reduced = svd.fit_transform(features)
    return reduced


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


def resize_feature_maps(fmaps, resize):
    resized_all = []
    for image_idx in range(len(fmaps)):
        resized_single_image = []
        for index, feature_map in enumerate(fmaps[image_idx]):
            feature_map = cv2.resize(feature_map, resize)
            resized_single_image.append(feature_map)
        resized_all.append(resized_single_image)
    return resized_all


def get_info_from_input_dir_name(input_dir_name, seperator='_'):
    num_fmaps = None
    split = input_dir_name.split(seperator)
    for item in split:
        if 'layer' in item:
            num_fmaps = maps_per_layer[item]

    return num_fmaps


def get_size(pickle_paths, order='max'):
    size = [-1, -1] if order == 'max' else [float('inf'), float('inf')]
    for key in pickle_paths.keys():
        first_path = pickle_paths[key][0]
        with open(first_path, 'rb') as f:
            data = pickle.load(f)
            first_fmap = np.squeeze(data)[0]
            _size = np.shape(np.squeeze(first_fmap))
            for i in [0, 1]:
                if order == 'max':
                    if _size[i] > size[i]:
                        size[i] = _size[i]
                elif order == 'min':
                    if _size[i] < size[i]:
                        size[i] = _size[i]

    size = (size[1], size[0])
    return size


def get_output_path(output_dir, input_dirs, reduction_method, pca_n, batches):
    first_input_dir_name = Path(input_dirs[0]).stem
    stem = '{}-{}-{}-{}'.format(reduction_method, pca_n, first_input_dir_name, 'batches' if batches else 'simple')
    out_path = os.path.join(output_dir, stem)
    os.makedirs(out_path, exist_ok=True)
    return out_path


def read_fmaps(input_dir, keys, uniform_size='min', max_samples=None, label_suffix=''):
    pickle_paths = get_pickle_paths(input_dir, keys)
    size = get_size(pickle_paths, order=uniform_size)

    feature_maps = dict()
    labels = list()

    for key in keys:
        _feature_maps = list()
        # iterator = tqdm(pickle_paths[key][:max_samples], desc='Reading pickles from {}'.format(key))
        iterator = pickle_paths[key]
        if max_samples is not None:
            iterator = iterator[:max_samples]
        for path in iterator:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            data = resize_feature_maps(data, size)
            _feature_maps.append(data[0])
            labels.append(key + label_suffix)

        feature_maps[key] = _feature_maps

    return feature_maps, labels


def reduce_dimensions(feature_maps, reduction_method, pca_n=None):
    print('raw\t:\t{}'.format(np.shape(feature_maps)))
    features = np.reshape(feature_maps, [np.shape(feature_maps)[0], -1])
    print('reshape\t:\t{}'.format(np.shape(features)))
    if pca_n:
        features = pca(features, pca_n)
        print('pca\t:\t{}'.format(np.shape(features)))
    if reduction_method == 'tsne':
        features = TSNE().fit(features)  # openTSNE implementation
        # feature_vectors = TSNE(n_components=2).fit_transform(features)  # sklearn implementation
    elif reduction_method == 'umap':
        import umap
        features = umap.UMAP(random_state=42).fit_transform(features)
        # features = umap.UMAP().fit_transform(features)
    if reduction_method:
        print('{}\t:\t{}'.format(reduction_method, np.shape(features)))

    return features


def visualize_fmaps(input_dir, args, label_suffix=''):
    feature_maps, labels = read_fmaps(input_dir, keys=args.keys, uniform_size=args.uniform_size, max_samples=args.max_samples, label_suffix=label_suffix)
    if args.use_batches:
        features = None
        for key in args.keys:
            print("Reducing Dimensions for key: {}".format(key))
            _feature_maps = feature_maps[key]
            _features = reduce_dimensions(_feature_maps, args.reduction_method, args.pca_n)
            if features is None:
                features = _features
            else:
                features = np.concatenate([features, _features])
    else:
        feature_maps = np.concatenate([feature_maps['source'], feature_maps['target']])
        features = reduce_dimensions(feature_maps, args.reduction_method, args.pca_n)
    return features, labels


if __name__ == '__main__':
    start = timeit.default_timer()
    parser = argparse.ArgumentParser(description="Image Sampler")
    parser.add_argument("--input_dirs", type=str, default=['/home/malte/Downloads/GTA2Cityscapes_SourceOnly_FeatureMaps_layer3',
                                                           '/home/malte/Downloads/GTA2Cityscapes_AdvEnt_FeatureMaps_layer3'])
    parser.add_argument("--out", type=str, default="./viz-output-simple-pure-umap", help="Path to output folder")
    parser.add_argument("--reduction_method", type=str, default='umap', choices=['tsne', 'umap'], help="type of low dim viz")
    parser.add_argument("--fmaps", type=str, default=None, choices=['single', 'mean'], help="show feature map visualization")
    parser.add_argument("--pca_n", type=int, default=100, help="Principal components to use for dim reduction")
    parser.add_argument("--uniform_size", type=str, default='min', choices=['max', 'min'], help="resize source and target fmaps to min or max of the two")
    parser.add_argument("--use_batches", type=bool, default=True, help="Whether to process the samples for each key individually")
    parser.add_argument("--keys", default=['source', 'target'], help="")
    parser.add_argument("--fada_labels", type=str, default=None, help="Path to FADA labels. None if not used.")
    parser.add_argument("--max_samples", type=int, default=500, help="Maximum Number of Images to use")
    args = parser.parse_args()
    output_path = get_output_path(args.out, args.input_dirs, args.reduction_method, args.pca_n, args.use_batches)
    print('Saving results at: {}'.format(output_path))

    if args.pca_n is not None:
        args.pca_n = min(args.max_samples*2, args.pca_n)  # Can't have more principal components than observations

    if len(args.input_dirs) == 1:
        features, labels = visualize_fmaps(args.input_dirs[0], args)
    else:
        print('\n====== Calculating Visualization for SourceOnly Model ======')
        features_so, labels_so = visualize_fmaps(args.input_dirs[0], args)
        print('\n====== Calculating Visualization for Adapted Model ======')
        features_adapt, labels_adapt = visualize_fmaps(args.input_dirs[1], args, label_suffix='_adapt')
        features = np.concatenate([features_so, features_adapt])
        labels = np.concatenate([labels_so, labels_adapt])

    plot(features, labels, colors=color_dict, s=10, save_path=os.path.join(output_path, 'plot.png'))

    with open(os.path.join(output_path, 'feats.pickle'), "wb") as f_out:
        pickle.dump({'features': features, 'labels': labels}, f_out)

    stop = timeit.default_timer()
    print('Program execution time: {:.2f}s'.format(stop - start))



