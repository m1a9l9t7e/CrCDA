import argparse
import os
import numpy as np
import pickle
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool

parser = argparse.ArgumentParser(description="Generate label stat info")
parser.add_argument("--datadir",
                    default="/home/malte/Downloads/GTA5",
                    help="path to load data",
                    type=str,
                    )
parser.add_argument("--d_list",
                    default=None,
                    help="path to list of images",
                    type=str,
                    )
parser.add_argument("--nprocs",
                    default=16,
                    help="Number of processes",
                    type=int,
                    )
parser.add_argument("--output_dir",
                    default="./label-info",
                    help="path to save label info",
                    type=str,
                    )
args = parser.parse_args()
os.makedirs(args.output_dir, exist_ok=True)
imgdir = os.path.join(args.datadir, 'images')
labdir = os.path.join(args.datadir, 'labels')

if args.d_list is None:
    labfiles = os.listdir(labdir)
else:
    with open(args.d_list) as f:
        content = f.readlines()
    labfiles = [x.strip() for x in content]

nprocs = args.nprocs
savedir = args.output_dir

ignore_label = 255
id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,  # map gta5 ids to new ids shared by cityscapes dataset
                 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

id_to_class_name = {
    0: 'unlabeled',
    1: 'ego vehicle',
    2: 'rectification border',
    3: 'out of roi',
    4: 'static',
    5: 'dynamic',
    6: 'ground',
    7: 'road',
    8: 'sidewalk',
    9: 'parking',
    10: 'rail track',
    11: 'building',
    12: 'wall',
    13: 'fence',
    14: 'guard rail',
    15: 'bridge',
    16: 'tunnel',
    17: 'pole',
    18: 'polegroup',
    19: 'traffic light',
    20: 'traffic sign',
    21: 'vegetation',
    22: 'terrain',
    23: 'sky',
    24: 'person',
    25: 'rider',
    26: 'car',
    27: 'truck',
    28: 'bus',
    29: 'caravan',
    30: 'trailer',
    31: 'train',
    32: 'motorcycle',
    33: 'bicycle',
    -1: 'license plate'
}

mapped_id_to_class_names = {id_to_trainid[_id] if _id in id_to_trainid.keys() else -1: class_name for _id, class_name in id_to_class_name.items()}


def generate_label_info():
    label_to_file = [[] for _ in range(len(id_to_trainid.keys()))]
    file_to_label = {e: [] for e in os.listdir(imgdir)}

    for labfile in tqdm(labfiles):
        label = np.unique(np.array(Image.open(os.path.join(labdir, labfile)), dtype=np.float32))
        for lab in label:
            if lab in id_to_trainid.keys():
                l = id_to_trainid[lab]
                label_to_file[l].append(labfile)
                file_to_label[labfile].append(l)

    return label_to_file, file_to_label


def _foo(i):
    label_to_file = [[] for _ in range(len(id_to_trainid.keys()))]
    file_to_label = dict()
    file_to_label_soft = dict()
    labfile = labfiles[i]
    file_to_label[labfile] = []
    file_to_label_soft[labfile] = []
    label_arr = np.array(Image.open(os.path.join(labdir, labfile)), dtype=np.float32)
    classes, class_counts = np.unique(label_arr, return_counts=True)
    for _class in id_to_trainid.keys():
        if _class in classes:
            l = id_to_trainid[_class]
            label_to_file[l].append(labfile)
            file_to_label[labfile].append(l)
            file_to_label_soft[labfile].append(class_counts[list(classes).index(_class)])
        else:
            file_to_label_soft[labfile].append(0)

    return label_to_file, file_to_label, file_to_label_soft


def main():
    label_to_file = [[] for _ in range(len(id_to_trainid.keys()))]
    file_to_label = {e: [] for e in labfiles}
    file_to_label_soft = {e: [] for e in labfiles}

    if nprocs == 1:
        label_to_file, file_to_label = generate_label_info()
    else:
        with Pool(nprocs) as p:
            r = list(tqdm(p.imap(_foo, range(len(labfiles))), total=len(labfiles)))
        for l2f, f2l, f2l_soft in r:
            for lab in range(len(l2f)):
                label_to_file[lab].extend(l2f[lab])
            for fname in f2l.keys():
                file_to_label[fname].extend(f2l[fname])
                file_to_label_soft[fname].extend(f2l_soft[fname])

    for idx, label in enumerate(label_to_file):
        print('Number of images with label {} ({}): {}'.format(idx, mapped_id_to_class_names[idx], len(label)))

    fname_index_to_soft_label = {index: file_to_label_soft[key] for index, key in enumerate(file_to_label_soft.keys())}

    with open(os.path.join(savedir, 'gtav_label_info.pickle'), 'wb') as f:
        pickle.dump((label_to_file, file_to_label, file_to_label_soft, fname_index_to_soft_label), f)


if __name__ == "__main__":
    main()