# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import os
import sys
import copy
import math
import shutil
import random
import argparse

import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from skimage import io
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from skimage.feature import peak_local_max

from torch.utils.data import DataLoader

from core.networks import *
from core.datasets import *

from tools.general.io_utils import *
from tools.general.time_utils import *
from tools.general.json_utils import *

from tools.ai.log_utils import *
from tools.ai.optim_utils import *
from tools.ai.torch_utils import *

from tools.ai.augment_utils import *
from tools.ai.randaugment import *


parser = argparse.ArgumentParser()

###############################################################################
# Dataset
###############################################################################
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--data_dir', default='../VOCtrainval_11-May-2012/', type=str)

###############################################################################
# Network
###############################################################################
parser.add_argument('--architecture', default='resnet50', type=str)
###############################################################################
# Hyperparameter
###############################################################################
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--max_epoch', default=15, type=int)
parser.add_argument('--warmup_epoch', default=0, type=int)

parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--wd', default=1e-4, type=float)
parser.add_argument('--nesterov', default=True, type=str2bool)

parser.add_argument('--image_size', default=512, type=int)
parser.add_argument('--min_image_size', default=320, type=int)
parser.add_argument('--max_image_size', default=640, type=int)

parser.add_argument('--print_ratio', default=0.1, type=float)

parser.add_argument('--tag', default='', type=str)
parser.add_argument('--augment', default='', type=str)

# For Puzzle-CAM
parser.add_argument('--num_pieces', default=4, type=int)
parser.add_argument('--h_co', default=250, type=int)
parser.add_argument('--w_co', default=250, type=int)

# 'cl_pcl'
# 'cl_re'
# 'cl_conf'
# 'cl_pcl_re'
# 'cl_pcl_re_conf'
parser.add_argument('--loss_option', default='cl_pcl_re', type=str)

parser.add_argument('--level', default='feature', type=str) 

parser.add_argument('--re_loss', default='L1_Loss', type=str)  # 'L1_Loss', 'L2_Loss'
parser.add_argument('--re_loss_option', default='masking', type=str)  # 'none', 'masking', 'selection'

# parser.add_argument('--branches', default='0,0,0,0,0,1', type=str)

parser.add_argument('--alpha_re', default=1.0, type=float, help='weight for re_loss')
parser.add_argument('--alpha_schedule_re', default=0.50, type=float, help='decreasing rate of weight for re_loss')
parser.add_argument('--alpha_p', default=1.0, type=float, help='weight for p_loss')
parser.add_argument('--alpha_ema', default=0.99, type=float, help='weight for ema')


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


if __name__ == '__main__':
    ###################################################################################
    # Arguments
    ###################################################################################
    args = parser.parse_args()

    log_dir = create_directory('./experiments/logs/')
    data_dir = create_directory('./experiments/data/')
    model_dir = create_directory('./experiments/models/')
    tensorboard_dir = create_directory('./experiments/tensorboards/{}/'.format(args.tag))

    exp_time = time.strftime("%y.%m.%d-%H%M%S")
    log_path = log_dir + exp_time + '{}.txt'.format(args.tag)
    data_path = data_dir + exp_time + '{}.json'.format(args.tag)
    model_path = model_dir + exp_time + '{}.pth'.format(args.tag)
    best_model_path = model_dir + 'best_' + '{}.pth'.format(args.tag)
    last_model_path = model_dir + 'last_' + '{}.pth'.format(args.tag)

    set_seed(args.seed)
    log_func = lambda string='': log_print(string, log_path)
    
    log_func('[i] {}'.format(args.tag))
    log_func()

    ###################################################################################
    # Transform, Dataset, DataLoader
    ###################################################################################

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    normalize_fn = Normalize(imagenet_mean, imagenet_std)
    
    train_transforms = [
        RandomResize(args.min_image_size, args.max_image_size),
        RandomHorizontalFlip(),
    ]
    if 'colorjitter' in args.augment:
        train_transforms.append(transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1))
    
    if 'randaugment' in args.augment:
        train_transforms.append(RandAugmentMC(n=2, m=10))

    train_transform = transforms.Compose(train_transforms + \
        [
            Normalize(imagenet_mean, imagenet_std),
            RandomCrop(args.image_size),
            Transpose()
        ]
    )
    test_transform = transforms.Compose([
        Normalize_For_Segmentation(imagenet_mean, imagenet_std),
        Top_Left_Crop_For_Segmentation(args.image_size),
        Transpose_For_Segmentation()
    ])
    
    meta_dic = read_json('./data/VOC_2012.json')
    class_names = np.asarray(meta_dic['class_names'])
    
    train_dataset = VOC_Dataset_For_Classification(args.data_dir, 'train_aug', train_transform)
    train_dataset_for_seg = VOC_Dataset_For_Testing_CAM(args.data_dir, 'train', test_transform)
    valid_dataset_for_seg = VOC_Dataset_For_Testing_CAM(args.data_dir, 'val', test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=True)
    train_loader_for_seg = DataLoader(train_dataset_for_seg, batch_size=args.batch_size, num_workers=1, drop_last=True)
    valid_loader_for_seg = DataLoader(valid_dataset_for_seg, batch_size=args.batch_size, num_workers=1, drop_last=True)

    log_func('[i] mean values is {}'.format(imagenet_mean))
    log_func('[i] std values is {}'.format(imagenet_std))
    log_func('[i] The number of class is {}'.format(meta_dic['classes']))
    log_func('[i] train_transform is {}'.format(train_transform))
    log_func('[i] test_transform is {}'.format(test_transform))
    log_func()

    val_iteration = len(train_loader)
    log_iteration = int(val_iteration * args.print_ratio)
    max_iteration = args.max_epoch * val_iteration

    log_func('[i] log_iteration : {:,}'.format(log_iteration))
    log_func('[i] val_iteration : {:,}'.format(val_iteration))
    log_func('[i] max_iteration : {:,}'.format(max_iteration))
    
    ###################################################################################
    # Network
    ###################################################################################
    model = Classifier(args.architecture, meta_dic['classes'])
    param_groups = model.get_parameter_groups(print_fn=None)

    gap_fn = model.global_average_pooling_2d

    model = model.cuda()
    model.train()

    log_func('[i] Architecture is {}'.format(args.architecture))
    log_func('[i] Total Params: %.2fM' % (calculate_parameters(model)))
    log_func()

    try:
        use_gpu = os.environ['CUDA_VISIBLE_DEVICES']
    except KeyError:
        use_gpu = '1'

    model_ema = Classifier(args.architecture, meta_dic['classes'])
    model_ema = model_ema.cuda()
    model_ema.train()
    for p in model_ema.parameters():
        p.detach_()

    the_number_of_gpu = len(use_gpu.split(','))
    if the_number_of_gpu > 1:
        log_func('[i] the number of gpu : {}'.format(the_number_of_gpu))
        model = nn.DataParallel(model)
        model_ema = nn.DataParallel(model_ema)

    load_model_fn = lambda: load_model(model, model_path, parallel=the_number_of_gpu > 1)
    save_model_fn = lambda: save_model(model, model_path, parallel=the_number_of_gpu > 1)
    
    ###################################################################################
    # Loss, Optimizer
    ###################################################################################
    class_loss_fn = nn.MultiLabelSoftMarginLoss(reduction='none').cuda()

    if args.re_loss == 'L1_Loss':
        re_loss_fn = L1_Loss
    else:
        re_loss_fn = L2_Loss

    log_func('[i] The number of pretrained weights : {}'.format(len(param_groups[0])))
    log_func('[i] The number of pretrained bias : {}'.format(len(param_groups[1])))
    log_func('[i] The number of scratched weights : {}'.format(len(param_groups[2])))
    log_func('[i] The number of scratched bias : {}'.format(len(param_groups[3])))
    
    optimizer = PolyOptimizer([
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wd},
        {'params': param_groups[1], 'lr': 2*args.lr, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10*args.lr, 'weight_decay': args.wd},
        {'params': param_groups[3], 'lr': 20*args.lr, 'weight_decay': 0},
    ], lr=args.lr, momentum=0.9, weight_decay=args.wd, warmup_epoch=args.warmup_epoch, max_step=max_iteration, nesterov=args.nesterov)
    
    #################################################################################################
    # Train
    #################################################################################################
    data_dic = {
        'train': [],
        'validation': []
    }

    train_timer = Timer()
    eval_timer = Timer()
    
    train_meter = Average_Meter(['loss', 'class_loss', 'p_class_loss', 're_loss', 'conf_loss'])
    
    best_train_mIoU = -1
    best_map = -1
    thresholds = list(np.arange(0.10, 0.50, 0.05))

    writer = SummaryWriter(tensorboard_dir)
    train_iterator = Iterator(train_loader)

    loss_option = args.loss_option.split('_')

    for iteration in range(max_iteration):

        images, labels = train_iterator.get()
        images, labels = images.cuda(), labels.cuda()
        # images:[32,3,512,512]  labels:[32,20]

        ###############################################################################
        # Normal
        ###############################################################################

        logits, features = model(images, with_cam=True)
        # with attention

        ###############################################################################
        # CAMs tensor -> numpy.array
        ###############################################################################

        cams = make_cam(features)
        cams_np = get_numpy_from_tensor(cams)               # cams_np: [32,20,32,32]

        ###############################################################################
        # Tiled_coordinates: [32, ..., 2]  img_choos_class: [32, ...]
        ###############################################################################
        tiled_coordinates = []
        img_choos_class = []
        for m in range(len(cams_np)):
            coor_per_image = []
            class_per_image = []
            for n in range(len(cams_np[0])):  # loop  < 20 times
                if labels[m][n] == 1:
                    class_per_image.append(n)
                    cam_np = cams_np[m][n][:][:]
                    cam_np = cv2.resize(cam_np, (args.image_size, args.image_size),
                                        interpolation=cv2.INTER_LINEAR)  # cam_np : [512,512]

                    max_list = []
                    local_max_coordinates = peak_local_max(cam_np, min_distance=20)
                    max_index = np.unravel_index(np.argmax(cam_np), cam_np.shape)

                    if len(local_max_coordinates) == 0:
                        coor_per_image.append([256, 256])
                        continue

                    for q in range(len(local_max_coordinates)):
                        max_list.append(cam_np[local_max_coordinates[q][0]][local_max_coordinates[q][1]])

                    max_arr = np.array(max_list)
                    index = np.argsort(-max_arr)

                    k = 5
                    coor_0 = 0
                    coor_1 = 0
                    num = 0

                    if len(max_list) <= k:
                        sum = 0
                        for z in range(len(max_list)):
                            sum += max_list[z]
                        for l in range(len(max_list)):
                            coor_0 += int(local_max_coordinates[index[l]][0] * max_list[index[l]] / sum)
                            coor_1 += int(local_max_coordinates[index[l]][1] * max_list[index[l]] / sum)
                        # coor_0 = coor_0 // len(max_list)
                        # coor_1 = coor_1 // len(max_list)
                        if coor_1 == 0 or coor_0 == 0:
                            avg_index = [256, 256]
                        else:
                            avg_index = [coor_0, coor_1]
                    else:
                        sum = 0
                        for z in range(k):
                            sum += max_list[index[z]]
                        for l in range(k):
                            coor_0 += int(local_max_coordinates[index[l]][0] * max_list[index[l]] / sum)
                            coor_1 += int(local_max_coordinates[index[l]][1] * max_list[index[l]] / sum)
                        # coor_0 = coor_0 // k
                        # coor_1 = coor_1 // k
                        if coor_1 == 0 or coor_0 == 0:
                            avg_index = [256, 256]
                        else:
                            avg_index = [coor_0, coor_1]

                    coor_per_image.append(avg_index)
            img_choos_class.append(class_per_image)
            tiled_coordinates.append(coor_per_image)

        ###############################################################################
        # Puzzle Module
        ###############################################################################

        h_per_patch = 256
        w_per_patch = 256
        cut_num = 0

        p_class_loss = 0
        re_loss = 0

        for i in range(len(tiled_coordinates)):  # loop 32 times
            for j in range(len(tiled_coordinates[i])):  # loop cut number
                cut_num += 1

        for i in range(len(tiled_coordinates)):
            for j in range(len(tiled_coordinates[i])):
                image_per_cut = []
                titled_index = tiled_coordinates[i][j]

                p1 = images[i, :, :titled_index[0], :titled_index[1]]         #
                p1_new = p1.unsqueeze(0)
                p1_new = torch.nn.functional.interpolate(p1_new, size=[h_per_patch, w_per_patch], scale_factor=None,
                                                         mode='nearest', align_corners=None)
                # p1_new: [1, 3, 256, 256]
                image_per_cut.append(p1_new)

                p2 = images[i, :, :titled_index[0], titled_index[1]:]
                p2_new = p2.unsqueeze(0)
                p2_new = torch.nn.functional.interpolate(p2_new, size=[h_per_patch, w_per_patch], scale_factor=None,
                                                         mode='nearest', align_corners=None)
                image_per_cut.append(p2_new)

                p3 = images[i, :, titled_index[0]:, :titled_index[1]]
                p3_new = p3.unsqueeze(0)
                p3_new = torch.nn.functional.interpolate(p3_new, size=[h_per_patch, w_per_patch], scale_factor=None,
                                                         mode='nearest', align_corners=None)
                image_per_cut.append(p3_new)

                p4 = images[i, :, titled_index[0]:, titled_index[1]:]
                p4_new = p4.unsqueeze(0)
                p4_new = torch.nn.functional.interpolate(p4_new, size=[h_per_patch, w_per_patch], scale_factor=None,
                                                         mode='nearest', align_corners=None)
                image_per_cut.append(p4_new)

                image_per_cut = torch.cat(image_per_cut, dim=0)                # image_per_cut: [4, 3, 256, 256]

                _, tiled_features = model_ema(image_per_cut, with_cam=True)        # tiled_features: [4, 20, 16, 16]

                features_list = list(torch.split(tiled_features, 1))           # item in features_list: [1, 20, 16, 16]

                merge_index = []
                merge_index.append(titled_index[0] // 16)
                merge_index.append(titled_index[1] // 16)

                if merge_index[0] == 0:
                    merge_index[0] = 1
                if merge_index[1] == 0:
                    merge_index[1] = 1

                # Assume merge_index: [ 3, 7]
                # Upper left: [1, 20, 3, 7]
                Upper_left = torch.nn.functional.interpolate(features_list[0], size=[merge_index[0], merge_index[1]], scale_factor=None,
                                                         mode='nearest', align_corners=None)
                # Upper right: [1, 20, 3, 25]
                Upper_right = torch.nn.functional.interpolate(features_list[1], size=[merge_index[0], 32-merge_index[1]],
                                                             scale_factor=None,
                                                             mode='nearest', align_corners=None)
                # Bottom left: [1, 20, 25, 7]
                Bottom_left = torch.nn.functional.interpolate(features_list[2], size=[32-merge_index[0], merge_index[1]],
                                                             scale_factor=None,
                                                             mode='nearest', align_corners=None)
                # Bottom right: [1, 20, 25, 25]
                Bottom_right = torch.nn.functional.interpolate(features_list[3], size=[32-merge_index[0], 32-merge_index[1]],
                                                             scale_factor=None,
                                                             mode='nearest', align_corners=None)

                # Up_half: [1, 20, 3, 32]
                Up_half = torch.cat([Upper_left, Upper_right], dim=3)

                # Bottom_half: [1, 20, 25, 32]
                Bottom_half = torch.cat([Bottom_left, Bottom_right], dim=3)

                # rel_features: [1, 20, 32, 32]
                re_features = torch.cat([Up_half, Bottom_half], dim=2)

                mask = torch.zeros(20)
                mask[img_choos_class[i][j]] = 1
                mask = mask.cuda()
                mask_1 = mask.unsqueeze(0)
                mask_2 = mask.unsqueeze(1).unsqueeze(2)

                p_class_loss += class_loss_fn(gap_fn(re_features), mask_1).mean()

                re_loss_p = re_loss_fn(features[i], re_features.squeeze(0)) * mask_2
                re_loss += re_loss_p.mean()

        p_class_loss = p_class_loss / cut_num

        re_loss = re_loss / cut_num

        class_loss = class_loss_fn(logits, labels).mean()

        conf_loss = torch.zeros(1).cuda()

        alpha_re = min(args.alpha_re * iteration / (max_iteration * args.alpha_schedule_re), args.alpha_re)

        loss = class_loss + args.alpha_p * p_class_loss + alpha_re * re_loss + conf_loss

        #################################################################################################

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        update_ema_variables(model, model_ema, args.alpha_ema, iteration)

        train_meter.add({
            'loss': loss.item(),
            'class_loss': class_loss.item(),
            'p_class_loss': p_class_loss.item(),
            're_loss': re_loss.item(),
            'conf_loss': conf_loss.item(),
        })

        #################################################################################################
        # For Log
        #################################################################################################
        if (iteration + 1) % log_iteration == 0:
            loss, class_loss, p_class_loss, re_loss, conf_loss = train_meter.get(clear=True)
            learning_rate = float(get_learning_rate_from_optimizer(optimizer))

            data = {
                'iteration': iteration + 1,
                'learning_rate': learning_rate,
                'loss': loss,
                'class_loss': class_loss,
                'p_class_loss': p_class_loss,
                're_loss': re_loss,
                'conf_loss': conf_loss,
                'time': train_timer.tok(clear=True),
            }
            data_dic['train'].append(data)
            write_json(data_path, data_dic)

            log_func('[i] \
                        iteration={iteration:,}, \
                        learning_rate={learning_rate:.4f}, \
                        loss={loss:.4f}, \
                        class_loss={class_loss:.4f}, \
                        p_class_loss={p_class_loss:.4f}, \
                        re_loss={re_loss:.4f}, \
                        conf_loss={conf_loss:.4f}, \
                        time={time:.0f}sec'.format(**data)
                     )

            writer.add_scalar('Train/loss', loss, iteration)
            writer.add_scalar('Train/class_loss', class_loss, iteration)
            writer.add_scalar('Train/p_class_loss', p_class_loss, iteration)
            writer.add_scalar('Train/re_loss', re_loss, iteration)
            writer.add_scalar('Train/conf_loss', conf_loss, iteration)
            writer.add_scalar('Train/learning_rate', learning_rate, iteration)


        ################################################################################################
        # Evaluation
        ################################################################################################
        def evaluate(loader):
            model.eval()
            eval_timer.tik()

            with torch.no_grad():
                # length = len(loader)
                labels_total = []
                scores_total = []
                eval_class_loss = 0
                count = 0
                for step, (images, labels, gt_masks) in enumerate(loader):
                    images = images.cuda()
                    labels = labels.cuda()

                    scores, _ = model(images, with_cam=True)  # scores size: [32, 20]

                    count += 1
                    eval_class_loss += class_loss_fn(scores, labels).mean()

                    labels_total.append(labels)
                    scores_total.append(scores)

                labels_total = torch.cat(labels_total, dim=0)
                scores_total = torch.cat(scores_total, dim=0)
                eval_class_loss /= count

                map = 0
                for k in range(scores_total.size(1)):  # loop 20 times

                    score_per_class = scores_total[:, k]  # score size: [32, 1]
                    label_per_class = labels_total[:, k]  # label size: [32, 1]

                    _, indices = torch.sort(score_per_class, dim=0, descending=True)

                    pos_count = 0.
                    total_count = 0.
                    precision_at_i = 0.

                    for i in indices:  # loop .... times
                        label = label_per_class[i]
                        if label == 1:
                            pos_count += 1
                        total_count += 1
                        if label == 1:
                            precision_at_i += pos_count / total_count
                    precision_at_i /= pos_count
                    map += precision_at_i
                map /= scores_total.size(1)

            print(' ')
            model.train()
            return eval_class_loss, map


        if (iteration + 1) % val_iteration == 0:

            eval_class_loss, map = evaluate(valid_loader_for_seg)

            if best_map == -1 or best_map < map:
                best_map = map
                save_model(model, best_model_path, parallel=the_number_of_gpu > 1)
                log_func('[i] save best model')

            if iteration + 1 == max_iteration:
                save_model(model, last_model_path, parallel=the_number_of_gpu > 1)
                log_func('[i] save last model')

            data = {
                'iteration': iteration + 1,
                'map': map * 100,
                'best map': best_map * 100,
                'eval_class_loss': eval_class_loss.item(),
                'time': eval_timer.tok(clear=True),
            }

            data_dic['validation'].append(data)
            write_json(data_path, data_dic)

            log_func('[i] \
                            iteration={iteration:,}, \
                            map={map:.2f}%, \
                            best map={best map:.2f}%, \
                            eval_class_loss={eval_class_loss:.4f}, \
                            time={time:.0f}sec'.format(**data)
                     )

            writer.add_scalar('Evaluation/map', map, iteration)
            writer.add_scalar('Evaluation/best map', best_map, iteration)
            writer.add_scalar('Evaluation/eval class loss', eval_class_loss, iteration)

    write_json(data_path, data_dic)
    writer.close()
