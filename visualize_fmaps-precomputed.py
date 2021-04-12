import argparse
import os
import numpy as np
from util import stitch_images, plot, stitch_horizontal, stitch_vertical
import pickle
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


if __name__ == '__main__':
    start = timeit.default_timer()
    parser = argparse.ArgumentParser(description="Image Sampler")
    parser.add_argument("--out", type=str, default="./viz-output-precomputed", help="Path to output folder")
    args = parser.parse_args()

    with open('/home/malte/PycharmProjects/Masterarbeit/CrCDA/pickles/umap-feats-SO.pickle', 'rb') as f:
        features_so = pickle.load(f)

    with open('/home/malte/PycharmProjects/Masterarbeit/CrCDA/pickles/umap-feats-AD.pickle', "rb") as f_in:
        features_adapt = pickle.loads(f_in.read())

    labels_num = np.concatenate([np.zeros([int(np.shape(features_so)[0]/2)]), np.ones([int(np.shape(features_so)[0]/2)])])
    labels_so = ['source' if label < 0.5 else 'target' for label in labels_num]
    labels_adapt = ['source_adapt' if label < 0.5 else 'target_adapt' for label in labels_num]

    features = np.concatenate([features_so, features_adapt])
    labels = np.concatenate([labels_so, labels_adapt])

    output_path = args.out
    print('Saving results at: {}'.format(output_path))

    plot(features, labels, colors=color_dict, s=10, save_path=os.path.join(output_path, 'plot.png'))

    with open(os.path.join(output_path, 'feats.pickle'), "wb") as f_out:
        pickle.dump({'features': features, 'labels': labels}, f_out)

    stop = timeit.default_timer()
    print('Program execution time: {:.2f}s'.format(stop - start))



