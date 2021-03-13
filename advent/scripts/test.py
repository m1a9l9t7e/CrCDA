# --------------------------------------------------------
# AdvEnt training
# Copyright (c) 2019 valeo.ai
#
# Written by Tuan-Hung Vu
# --------------------------------------------------------
import argparse
import os
import os.path as osp
import pprint
import warnings

from torch import nn
from torch.utils import data
from tensorboardX import SummaryWriter

from advent.dataset.gta5 import GTA5DataSet
from advent.model.deeplabv2 import get_deeplab_v2
from advent.dataset.cityscapes import CityscapesDataSet
from advent.domain_adaptation.config import cfg, cfg_from_file
from advent.domain_adaptation.eval_UDA import evaluate_domain_adaptation


warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore")


def get_arguments():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Code for evaluation")
    parser.add_argument('--cfg', type=str, default=None,
                        help='optional config file', )
    parser.add_argument("--exp-suffix", type=str, default=None,
                        help="optional experiment suffix")
    return parser.parse_args()


def main(config_file, exp_suffix, fixed_test_size=True):
    # LOAD ARGS
    assert config_file is not None, 'Missing cfg file'
    cfg_from_file(config_file)
    # auto-generate exp name if not specified
    if cfg.EXP_NAME == '':
        cfg.EXP_NAME = f'{cfg.SOURCE}2{cfg.TARGET}_{cfg.TRAIN.MODEL}_{cfg.TRAIN.DA_METHOD}'
    if exp_suffix:
        cfg.EXP_NAME += f'_{exp_suffix}'
    # auto-generate snapshot path if not specified
    if cfg.TEST.SNAPSHOT_DIR[0] == '':
        cfg.TEST.SNAPSHOT_DIR[0] = osp.join(cfg.EXP_ROOT_SNAPSHOT, cfg.EXP_NAME)
        os.makedirs(cfg.TEST.SNAPSHOT_DIR[0], exist_ok=True)

    # init tensorboard writer
    log_path = osp.join(cfg.EXP_ROOT_EVAL, cfg.EXP_NAME)
    writer = SummaryWriter(log_path)
    writer.add_text("Info", str(cfg))

    print("Evaluating experiment {}, with weights at {}".format(cfg.EXP_NAME, cfg.TEST.SNAPSHOT_DIR))
    print("Writing tensorboard results to {}".format(log_path))
    print('Using config:')
    pprint.pprint(cfg)
    # load models
    models = []
    n_models = len(cfg.TEST.MODEL)
    if cfg.TEST.MODE == 'best':
        assert n_models == 1, 'Not yet supported'
    for i in range(n_models):
        if cfg.TEST.MODEL[i] == 'DeepLabv2':
            model = get_deeplab_v2(num_classes=cfg.NUM_CLASSES,
                                   multi_level=cfg.TEST.MULTI_LEVEL[i])
        else:
            raise NotImplementedError(f"Not yet supported {cfg.TEST.MODEL[i]}")
        models.append(model)

    if os.environ.get('ADVENT_DRY_RUN', '0') == '1':
        return

    # dataloader source
    test_dataset_source = GTA5DataSet(root=cfg.DATA_DIRECTORY_SOURCE,
                                      list_path=cfg.DATA_LIST_SOURCE,
                                      set=cfg.TEST.SET_SOURCE,
                                      crop_size=cfg.TEST.INPUT_SIZE_SOURCE,
                                      mean=cfg.TEST.IMG_MEAN)
    test_loader_source = data.DataLoader(test_dataset_source,
                                         batch_size=cfg.TEST.BATCH_SIZE_SOURCE,
                                         num_workers=cfg.NUM_WORKERS,
                                         shuffle=False,
                                         pin_memory=True)

    # eval source
    interp_source = None
    if fixed_test_size:
        interp_source = nn.Upsample(size=(cfg.TEST.OUTPUT_SIZE_SOURCE[1], cfg.TEST.OUTPUT_SIZE_SOURCE[0]), mode='bilinear', align_corners=True)

    print("<==:MODE_CHANGE:SOURCE_EVAL:==>")
    evaluate_domain_adaptation(models, test_loader_source, cfg, descriptor='mIoU_source', interp=interp_source, tensorboard_writer=writer, verbose=False)

    # dataloader target
    test_dataset_target = CityscapesDataSet(root=cfg.DATA_DIRECTORY_TARGET,
                                            list_path=cfg.DATA_LIST_TARGET,
                                            set=cfg.TEST.SET_TARGET,
                                            info_path=cfg.TEST.INFO_TARGET,
                                            crop_size=cfg.TEST.INPUT_SIZE_TARGET,
                                            mean=cfg.TEST.IMG_MEAN,
                                            labels_size=cfg.TEST.OUTPUT_SIZE_TARGET)
    test_loader_target = data.DataLoader(test_dataset_target,
                                         batch_size=cfg.TEST.BATCH_SIZE_TARGET,
                                         num_workers=cfg.NUM_WORKERS,
                                         shuffle=False,
                                         pin_memory=True)

    # eval target
    interp_target = None
    if fixed_test_size:
        interp_target = nn.Upsample(size=(cfg.TEST.OUTPUT_SIZE_TARGET[1], cfg.TEST.OUTPUT_SIZE_TARGET[0]), mode='bilinear', align_corners=True)

    print("<==:MODE_CHANGE:TARGET_EVAL:==>")
    evaluate_domain_adaptation(models, test_loader_target, cfg, descriptor='mIoU_target', interp=interp_target, tensorboard_writer=writer)


if __name__ == '__main__':
    args = get_arguments()
    print('Called with args:')
    print(args)
    main(args.cfg, args.exp_suffix)
