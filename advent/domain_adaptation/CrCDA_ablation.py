# --------------------------------------------------------
# Domain adpatation training
# Copyright (c) 2019 valeo.ai
#
# Written by Tuan-Hung Vu
# --------------------------------------------------------
import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch import nn
from torchvision.utils import make_grid
from advent.domain_adaptation.eval_UDA import display_stats
from advent.model.discriminator import get_fc_discriminator
from advent.utils.func import adjust_learning_rate, adjust_learning_rate_discriminator, fast_hist, per_class_iu
from advent.utils.func import loss_calc, bce_loss
from advent.utils.loss import entropy_loss, entropy_loss_with_regularization
from advent.utils.func import prob_2_entropy
from advent.utils.viz_segmask import colorize_mask
from advent.model.grl import LambdaWrapper
from tqdm import trange


def train_crcda(model, trainloader, targetloader, cfg, testloader=None):
    ''' UDA training with CrCDA '''
    # Create the model and start the training.
    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    input_size_target = cfg.TRAIN.INPUT_SIZE_TARGET
    device = cfg.GPU_ID
    num_classes = cfg.NUM_CLASSES
    viz_tensorboard = os.path.exists(cfg.TRAIN.TENSORBOARD_LOGDIR)
    if viz_tensorboard:
        writer = SummaryWriter(log_dir=cfg.TRAIN.TENSORBOARD_LOGDIR)

    # SEGMNETATION NETWORK
    model.train()
    model.to(device)
    cudnn.benchmark = True
    cudnn.enabled = True

    # DISCRIMINATOR NETWORK
    # seg maps, i.e. output, level
    d_main = get_fc_discriminator(num_classes=num_classes)
    d_main.train()
    d_main.to(device)

    # OPTIMIZERS
    # segnet's optimizer
    optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                          lr=cfg.TRAIN.LEARNING_RATE,
                          momentum=cfg.TRAIN.MOMENTUM,
                          weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    # discriminators' optimizers
    optimizer_d_main = optim.Adam(d_main.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
                                  betas=(0.9, 0.99))

    # interpolate output segmaps
    interp = nn.Upsample(size=(input_size_source[1], input_size_source[0]), mode='bilinear',
                         align_corners=True)
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear',
                                align_corners=True)

    # labels for adversarial training
    source_label = 0
    target_label = 1
    trainloader_iter = enumerate(trainloader)
    targetloader_iter = enumerate(targetloader)

    t = trange(cfg.TRAIN.EARLY_STOP + 1, desc='Loss', leave=True)
    for i_iter in t:
        # reset optimizers
        optimizer.zero_grad()
        # optimizer_d_aux.zero_grad()
        optimizer_d_main.zero_grad()
        # adapt LR if needed
        adjust_learning_rate(optimizer, i_iter, cfg)
        # adjust_learning_rate_discriminator(optimizer_d_aux, i_iter, cfg)
        adjust_learning_rate_discriminator(optimizer_d_main, i_iter, cfg)

        # UDA Training
        # only train segnet. Don't accumulate grads in disciminators
        for param in d_main.parameters():
            param.requires_grad = False

        # -------------- SOURCE FLOW --------------
        images_source, loss_c2, loss_c3, loss_seg, pred_src_layout, pred_src_seg = source_flow(cfg, device, interp, model, trainloader_iter)

        # -------------- TARGET FLOW --------------
        images, loss_adv_trg, loss_ent, loss_ent2, loss_ent3, pred_trg_layout, pred_trg_seg = target_flow(cfg, d_main, device, interp_target, model,
                                                                                                          source_label, targetloader_iter, i_iter)

        # -------------- DISCRIMINATOR --------------
        loss_d = 0
        if cfg.TRAIN.USE_DISCRIMINATOR:
            loss_d = update_discriminator(d_main, optimizer, optimizer_d_main, pred_src_layout, pred_trg_layout, source_label, target_label)
        del pred_src_layout
        del pred_trg_layout

        optimizer.step()
        # --------------- LOGGING -----------------
        current_losses = {
            'loss_src_seg': loss_seg,
            'loss_src_c2': loss_c2,
            'loss_src_c3': loss_c3,
            'loss_trg_ent': loss_ent,
            'loss_trg_ent2': loss_ent2,
            'loss_trg_ent3': loss_ent3,
            'loss_trg_adv': loss_adv_trg,
            'loss_d': loss_d  # this is only the d loss from training with target!
        }

        t.set_description(get_loss_string(current_losses, i_iter))
        logging(cfg, current_losses, d_main, device, i_iter, images, images_source, model, num_classes, pred_src_seg, pred_trg_seg, viz_tensorboard, writer, testloader=testloader)
        sys.stdout.flush()

        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter >= cfg.TRAIN.EARLY_STOP - 1:
            break


def source_flow(cfg, device, interp, model, trainloader_iter):
    _, batch = trainloader_iter.__next__()
    images_source, labels, patch_scale_labels, mini_patch_scale_labels, _, _ = batch
    pred_src_seg, pred_src_cr_mini, pred_src_cr = model(images_source.cuda(device))

    loss = 0
    pred_src_layout = []

    # L_seg
    loss_seg = 0
    if cfg.TRAIN.USE_SEG:
        pred_src_seg = interp(pred_src_seg)
        pred_src_layout.append(pred_src_seg)
        loss_seg = loss_calc(pred_src_seg, labels, device)
        loss += cfg.TRAIN.LAMBDA_SEG * loss_seg
        loss_seg = loss_seg.detach()
    # L_c2
    loss_c2 = 0
    if cfg.TRAIN.USE_MINI_PATCH:
        pred_src_cr_mini = interp(pred_src_cr_mini)
        pred_src_layout.append(pred_src_cr_mini)
        loss_c2 = loss_calc(pred_src_cr_mini, mini_patch_scale_labels, device)
        loss += cfg.TRAIN.LAMBDA_C2 * loss_c2
        loss_c2 = loss_c2.detach()
    # L_c3
    loss_c3 = 0
    if cfg.TRAIN.USE_PATCH:
        pred_src_cr = interp(pred_src_cr)
        pred_src_layout.append(pred_src_cr)
        loss_c3 = loss_calc(pred_src_cr, patch_scale_labels, device)
        loss += cfg.TRAIN.LAMBDA_C3 * loss_c3
        loss_c3 = loss_c3.detach()

    if cfg.TRAIN.USE_DISCRIMINATOR and len(pred_src_layout) > 0:
        pred_src_layout = concatenate_padding(pred_src_layout, device, padding_dimension=1, concat_dimension=0)

    if loss != 0:
        loss.backward()

    pred_src_layout = [tensor.detach() for tensor in pred_src_layout]
    return images_source, loss_c2, loss_c3, loss_seg, pred_src_layout, pred_src_seg.detach()


def target_flow(cfg, d_main, device, interp_target, model, source_label, targetloader_iter, i_iter):
    """ adversarial training with local minmax entropy loss and global alignment discriminator loss """
    _, batch = targetloader_iter.__next__()
    images_target, _, _, _ = batch
    lambda_wrapper = LambdaWrapper(lambda_=-1)
    pred_trg_seg, pred_trg_cr_mini, pred_trg_cr = model(images_target.cuda(device), grl_lambda=lambda_wrapper)

    loss = 0
    pred_trg_layout = []
    # L_ent
    loss_ent = 0
    if cfg.TRAIN.USE_SEG_ENT:
        pred_trg_seg = interp_target(pred_trg_seg)
        pred_trg_layout.append(pred_trg_seg)
        loss_ent = entropy_loss_with_regularization(F.softmax(pred_trg_seg), i_iter, cfg.TRAIN.ENT_REG_MAX_ITER)
        loss += cfg.TRAIN.LAMBDA_ENT * loss_ent
        loss_ent = loss_ent.detach()
    # L_ent2
    loss_ent2 = 0
    if cfg.TRAIN.USE_MINI_PATCH_ENT:
        pred_trg_cr_mini = interp_target(pred_trg_cr_mini)
        pred_trg_layout.append(pred_trg_cr_mini)
        loss_ent2 = entropy_loss_with_regularization(F.softmax(pred_trg_cr_mini), i_iter, cfg.TRAIN.ENT_REG_MAX_ITER)
        loss += cfg.TRAIN.LAMBDA_ENT2 * loss_ent2
        loss_ent2 = loss_ent2.detach()
    # L_ent2
    loss_ent3 = 0
    if cfg.TRAIN.USE_PATCH_ENT:
        pred_trg_cr = interp_target(pred_trg_cr)
        pred_trg_layout.append(pred_trg_cr)
        loss_ent3 = entropy_loss_with_regularization(F.softmax(pred_trg_cr), i_iter, cfg.TRAIN.ENT_REG_MAX_ITER)
        loss += cfg.TRAIN.LAMBDA_ENT3 * loss_ent3
        loss_ent3 = loss_ent3.detach()

    loss = -loss  # This way, seg heads maximize entropy, and feature extractor minimizes it

    if loss != 0:
        # keep grad graph for application of discriminator adv loss with different grl lambda
        loss.backward(retain_graph=(cfg.TRAIN.USE_DISCRIMINATOR and len(pred_trg_layout) > 0))

    loss_adv_trg = 0
    if cfg.TRAIN.USE_DISCRIMINATOR and len(pred_trg_layout) > 0:
        pred_trg_layout = concatenate_padding(pred_trg_layout, device, padding_dimension=1, concat_dimension=0)
        d_out = d_main(F.softmax(pred_trg_layout))  # TODO: Softmax here?
        loss_adv_trg = bce_loss(d_out, source_label)
        loss += cfg.TRAIN.LAMBDA_ADV * loss_adv_trg
        lambda_wrapper.set_lambda(1)
        loss.backward()
        loss_adv_trg = loss_adv_trg.detach()

    return images_target, loss_adv_trg, loss_ent, loss_ent2, loss_ent3, pred_trg_layout, pred_trg_seg.detach()


def update_discriminator(d_main, optimizer, optimizer_d_main, pred_src_layout, pred_trg_layout, source_label, target_label):
    # Train discriminator network
    # enable training mode on discriminator networks
    for param in d_main.parameters():
        param.requires_grad = True
    # train with source
    pred_src = pred_src_layout.detach()
    d_out = d_main(F.softmax(pred_src))  # TODO: Softmax here?
    loss_d = bce_loss(d_out, source_label)
    loss_d = loss_d / 2
    loss_d.backward()
    # train with target
    pred_trg = pred_trg_layout.detach()
    d_out = d_main(F.softmax(pred_trg))  # TODO: Softmax here?
    loss_d = bce_loss(d_out, target_label)
    loss_d = loss_d / 2
    loss_d.backward()
    optimizer_d_main.step()
    return loss_d.detach()


def logging(cfg, current_losses, d_main, device, i_iter, images, images_source, model, num_classes, pred_src_seg, pred_trg_seg, viz_tensorboard, writer, testloader=None):
    if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
        # print('\ntaking snapshot ...')
        # print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
        snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
        torch.save(model.state_dict(), snapshot_dir / f'model_{i_iter}.pth')
        torch.save(d_main.state_dict(), snapshot_dir / f'model_{i_iter}_D_main.pth')
        if testloader is not None:
            eval_model(cfg, model, testloader, i_iter, writer, device)
    # Visualize with tensorboard
    if viz_tensorboard:
        log_losses_tensorboard(writer, current_losses, i_iter)

        if i_iter % cfg.TRAIN.TENSORBOARD_VIZRATE == cfg.TRAIN.TENSORBOARD_VIZRATE - 1:
            draw_in_tensorboard(writer, images, i_iter, pred_trg_seg, num_classes, 'T')
            draw_in_tensorboard(writer, images_source, i_iter, pred_src_seg, num_classes, 'S')


def get_loss_string(current_losses, i_iter):
    list_strings = []
    for loss_name, loss_value in current_losses.items():
        if loss_value != 0:
            list_strings.append(f'{loss_name} = {to_numpy(loss_value):.3f} ')
    full_string = ' '.join(list_strings)
    return f'iter = {i_iter} {full_string}'


def log_losses_tensorboard(writer, current_losses, i_iter):
    for loss_name, loss_value in current_losses.items():
        writer.add_scalar(f'data/{loss_name}', to_numpy(loss_value), i_iter)


def to_numpy(tensor):
    if isinstance(tensor, (int, float)):
        return tensor
    else:
        return tensor.data.cpu().numpy()


def concatenate_padding(tensors, device, padding_dimension=1, concat_dimension=0):
    new_list = []
    max_dim1 = 0
    for tensor in tensors:
        if np.shape(tensor)[padding_dimension] > max_dim1:
            max_dim1 = np.shape(tensor)[padding_dimension]

    for tensor in tensors:
        # tensor_size = list(np.shape(to_numpy(tensor)))
        tensor_size = list(np.shape(tensor))
        if tensor_size[padding_dimension] < max_dim1:
            tensor_size[padding_dimension] = max_dim1 - tensor_size[padding_dimension]
            new_list.append(torch.cat([tensor, torch.zeros(tensor_size).cuda(device)], padding_dimension))

    return torch.cat(new_list, concat_dimension)


def draw_in_tensorboard(writer, images, i_iter, pred_main, num_classes, type_):
    grid_image = make_grid(images[:3].clone().cpu().data, 3, normalize=True)
    writer.add_image(f'Image - {type_}', grid_image, i_iter)

    grid_image = make_grid(torch.from_numpy(np.array(colorize_mask(np.asarray(
        np.argmax(F.softmax(pred_main).cpu().data[0].numpy().transpose(1, 2, 0),
                  axis=2), dtype=np.uint8)).convert('RGB')).transpose(2, 0, 1)), 3,
                           normalize=False, range=(0, 255))
    writer.add_image(f'Prediction - {type_}', grid_image, i_iter)

    output_sm = F.softmax(pred_main).cpu().data[0].numpy().transpose(1, 2, 0)
    output_ent = np.sum(-np.multiply(output_sm, np.log2(output_sm)), axis=2,
                        keepdims=False)
    grid_image = make_grid(torch.from_numpy(output_ent), 3, normalize=True,
                           range=(0, np.log2(num_classes)))
    writer.add_image(f'Entropy - {type_}', grid_image, i_iter)


def eval_model(cfg, model, testloader, i_iter, writer, device, fixed_test_size=True, descriptor='target_miou', verbose=False, extra=True):
    # eval target
    interp_target = None
    if fixed_test_size:
        interp_target = nn.Upsample(size=(cfg.TEST.OUTPUT_SIZE_TARGET[1], cfg.TEST.OUTPUT_SIZE_TARGET[0]), mode='bilinear', align_corners=True)

    models = [model]
    interp = interp_target
    verbose = False
    assert len(cfg.TEST.RESTORE_FROM) == len(models), 'Number of models are not matched'
    hist = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES))
    print("\nEvaluating Model...")
    for index, batch in enumerate(testloader):
        image, label, _, name = batch
        if not fixed_test_size:
            interp = nn.Upsample(size=(label.shape[1], label.shape[2]), mode='bilinear', align_corners=True)
        with torch.no_grad():
            output = None
            for model, model_weight in zip(models, cfg.TEST.MODEL_WEIGHT):
                pred_main = model(image.cuda(device))[model.get_main_index()]
                output_ = interp(pred_main).cpu().data[0].numpy()
                if output is None:
                    output = model_weight * output_
                else:
                    output += model_weight * output_
            assert output is not None, 'Output is None'
            output = output.transpose(1, 2, 0)
            output = np.argmax(output, axis=2)
        label = label.numpy()[0]
        hist += fast_hist(label.flatten(), output.flatten(), cfg.NUM_CLASSES)
        sys.stdout.flush()
    inters_over_union_classes = per_class_iu(hist)
    miou = np.nanmean(inters_over_union_classes)
    if extra:
        print('\033[93m' + 'target val mIoU = ' + str(round(miou * 100, 2)) +'\033[0m\n')
    else:
        print(f'mIoU = \t{round(miou * 100, 2)}')
    if writer is not None:
        writer.add_scalar(descriptor, miou, i_iter)
    if verbose:
        display_stats(cfg, testloader.dataset.class_names, inters_over_union_classes)