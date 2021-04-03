import cv2
from skimage.exposure import exposure
from skimage.feature import hog
import matplotlib.pyplot as plt
import matplotlib
import numpy as np


def print_red(string):
    print('\u001B[31m' + string + '\u001B[0m')


def stitch_horizontal(image_top, image_bottom, padding):
    _shape = np.shape(image_top)
    image = np.concatenate([image_top, np.ones([padding, _shape[1], _shape[2]], dtype=np.uint8) * 255, image_bottom], axis=0)
    return image


def stitch_vertical(image_left, image_right, padding, padding_color=1):
    _shape = np.shape(image_left)
    image = np.concatenate([image_left, (np.zeros([_shape[0], padding, _shape[2]], dtype=np.uint8) + padding_color) * 255, image_right], axis=1)
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


def apply_padding(image, padding):
    vertical_padding = np.zeros((np.shape(image)[0], padding, 3))
    image = stitch_vertical(image, vertical_padding, 0)
    horizontal_padding = np.zeros((padding, np.shape(image)[1], 3))
    image = stitch_horizontal(image, horizontal_padding, 0)
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


def get_class_distribution(label, class_ids):
    distribution = []
    for class_id in class_ids:
        distribution.append(np.count(label, class_id))
    
    return distribution


def plot(x, y, save_path=None, ax=None, title=None, draw_legend=True, draw_centers=False, draw_cluster_labels=False, colors=None, legend_kwargs=None, label_order=None, **kwargs):
    if ax is None:
        _, ax = matplotlib.pyplot.subplots(figsize=(8, 8))

    if title is not None:
        ax.set_title(title)

    plot_params = {"alpha": kwargs.get("alpha", 0.9), "s": kwargs.get("s", 1)}

    # Create main plot
    if label_order is not None:
        assert all(np.isin(np.unique(y), label_order))
        classes = [l for l in label_order if l in np.unique(y)]
    else:
        classes = np.unique(y)
    if colors is None:
        default_colors = matplotlib.rcParams["axes.prop_cycle"]
        colors = {k: v["color"] for k, v in zip(classes, default_colors())}

    point_colors = list(map(colors.get, y))

    ax.scatter(x[:, 0], x[:, 1], c=point_colors, rasterized=True, **plot_params)

    # Plot mediods
    if draw_centers:
        centers = []
        for yi in classes:
            mask = yi == y
            centers.append(np.median(x[mask, :2], axis=0))
        centers = np.array(centers)

        center_colors = list(map(colors.get, classes))
        ax.scatter(
            centers[:, 0], centers[:, 1], c=center_colors, s=48, alpha=1, edgecolor="k"
        )

        # Draw mediod labels
        if draw_cluster_labels:
            for idx, label in enumerate(classes):
                ax.text(
                    centers[idx, 0],
                    centers[idx, 1] + 2.2,
                    label,
                    fontsize=kwargs.get("fontsize", 6),
                    horizontalalignment="center",
                )

    # Hide ticks and axis
    ax.set_xticks([]), ax.set_yticks([]), ax.axis("off")

    if draw_legend:
        legend_handles = [
            matplotlib.lines.Line2D(
                [],
                [],
                marker="s",
                color="w",
                markerfacecolor=colors[yi],
                ms=10,
                alpha=1,
                linewidth=0,
                label=yi,
                markeredgecolor="k",
            )
            for yi in classes
        ]
        legend_kwargs_ = dict(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False, )
        if legend_kwargs is not None:
            legend_kwargs_.update(legend_kwargs)
        ax.legend(handles=legend_handles, **legend_kwargs_)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    image_source_only = cv2.imread('so.png')
    image_adapt = cv2.imread('ad.png')
    image = stitch_vertical(image_source_only, image_adapt, padding=4, padding_color=0)
    cv2.imwrite('umap.png', image)
    cv2.imshow('win', image)
    cv2.waitKey(0)
