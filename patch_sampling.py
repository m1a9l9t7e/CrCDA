import argparse
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import cv2


class Sampler:

    samples = []

    """
    @:arg sample_shape has form (H x W x C)
    """
    def __init__(self, sample_shape, cluster_n=10):
        self.sample_shape = sample_shape
        self.samples = []

    def extract_samples(self, image):
        samples = sliding_window_view(image, self.sample_shape)[0::self.sample_shape[0], 0::self.sample_shape[1]]
        samples = samples.reshape([-1] + self.sample_shape)
        self.samples += samples
        return samples

    def cluster_samples(self):
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Image Sampler")
    parser.add_argument("--label_dir", type=str, default='./labels', help="Path to ground truth to extract samples from")
    parser.add_argument("--out", type=str, default="./samples", help="Path to output folder")
    args = parser.parse_args()

    # # numbers test
    # img = np.linspace((0, 10, 20, 30, 40, 50), (10, 20, 30, 40, 50, 60), 11)
    # sample_extractor = Sampler(sample_shape=[3, 3])
    # print(sample_extractor.extract_samples(img))

    # img test
    img = cv2.imread('./labels/00001.png')
    cv2.imshow('base', img)
    cv2.waitKey(0)
    sample_extractor = Sampler(sample_shape=[300, 600, img.shape[2]])
    _samples = sample_extractor.extract_samples(img)
    for _sample in _samples:
        cv2.imshow('sample', _sample)
        cv2.waitKey(0)
