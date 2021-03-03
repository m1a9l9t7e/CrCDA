import numpy as np

from advent.dataset.base_dataset import BaseDataset


class GTA5DataSetWithPatchLabels(BaseDataset):
    def __init__(self, root, list_path, set='all',
                 max_iters=None, crop_size=(321, 321), mean=(128, 128, 128)):
        super().__init__(root, list_path, set, max_iters, crop_size, None, mean)

        # map to cityscape's ids
        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

    def get_metadata(self, name):
        img_file = self.root / 'images' / name
        label_file = self.root / 'labels' / name
        patch_label_file = self.root / 'patch-labels' / name
        mini_patch_label_file = self.root / 'mini-patch-labels' / name
        return img_file, label_file, patch_label_file, mini_patch_label_file

    def __getitem__(self, index):
        img_file, label_file, patch_label_file, mini_patch_label_file, name = self.files[index]
        image = self.get_image(img_file)
        label = self.get_labels(label_file)
        patch_label = self.get_labels(patch_label_file)
        mini_patch_label = self.get_labels(mini_patch_label_file)
        # re-assign labels to match the format of Cityscapes
        label_copy = 255 * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
        image = self.preprocess(image)
        return image.copy(), label_copy.copy(), patch_label.copy(), mini_patch_label.copy(), np.array(image.shape), name
