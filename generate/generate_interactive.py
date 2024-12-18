# Copyright by UnSeg team
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import re
import os
import json
import argparse
import numpy as np
import time
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import random
from tqdm import tqdm
from typing import Dict, List, Tuple, Any
from utils.register_coco import register_all_coco_panoptic_annos_sem_seg

from segment_anything_training import sam_model_registry, noise_sam_model_registry, scratch_sam_model_registry
from segment_anything_training.modeling import TwoWayTransformer, MaskDecoder

from utils.dataloader import get_im_gt_name_dict, create_dataloaders, RandomHFlip, Resize, LargeScaleJitter, Resize_Padding, get_im_gt_name_dict_cityscapes
from utils.loss_mask import loss_masks
import utils.misc as misc
from torchvision.transforms import ToPILImage

coco_categories = [{"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "person"}, {"color": [119, 11, 32], "isthing": 1, "id": 2, "name": "bicycle"}, {"color": [0, 0, 142], "isthing": 1, "id": 3, "name": "car"}, {"color": [0, 0, 230], "isthing": 1, "id": 4, "name": "motorcycle"}, {"color": [106, 0, 228], "isthing": 1, "id": 5, "name": "airplane"}, {"color": [0, 60, 100], "isthing": 1, "id": 6, "name": "bus"}, {"color": [0, 80, 100], "isthing": 1, "id": 7, "name": "train"}, {"color": [0, 0, 70], "isthing": 1, "id": 8, "name": "truck"}, {"color": [0, 0, 192], "isthing": 1, "id": 9, "name": "boat"}, {"color": [250, 170, 30], "isthing": 1, "id": 10, "name": "traffic light"}, {"color": [100, 170, 30], "isthing": 1, "id": 11, "name": "fire hydrant"}, {"color": [220, 220, 0], "isthing": 1, "id": 13, "name": "stop sign"}, {"color": [175, 116, 175], "isthing": 1, "id": 14, "name": "parking meter"}, {"color": [250, 0, 30], "isthing": 1, "id": 15, "name": "bench"}, {"color": [165, 42, 42], "isthing": 1, "id": 16, "name": "bird"}, {"color": [255, 77, 255], "isthing": 1, "id": 17, "name": "cat"}, {"color": [0, 226, 252], "isthing": 1, "id": 18, "name": "dog"}, {"color": [182, 182, 255], "isthing": 1, "id": 19, "name": "horse"}, {"color": [0, 82, 0], "isthing": 1, "id": 20, "name": "sheep"}, {"color": [120, 166, 157], "isthing": 1, "id": 21, "name": "cow"}, {"color": [110, 76, 0], "isthing": 1, "id": 22, "name": "elephant"}, {"color": [174, 57, 255], "isthing": 1, "id": 23, "name": "bear"}, {"color": [199, 100, 0], "isthing": 1, "id": 24, "name": "zebra"}, {"color": [72, 0, 118], "isthing": 1, "id": 25, "name": "giraffe"}, {"color": [255, 179, 240], "isthing": 1, "id": 27, "name": "backpack"}, {"color": [0, 125, 92], "isthing": 1, "id": 28, "name": "umbrella"}, {"color": [209, 0, 151], "isthing": 1, "id": 31, "name": "handbag"}, {"color": [188, 208, 182], "isthing": 1, "id": 32, "name": "tie"}, {"color": [0, 220, 176], "isthing": 1, "id": 33, "name": "suitcase"}, {"color": [255, 99, 164], "isthing": 1, "id": 34, "name": "frisbee"}, {"color": [92, 0, 73], "isthing": 1, "id": 35, "name": "skis"}, {"color": [133, 129, 255], "isthing": 1, "id": 36, "name": "snowboard"}, {"color": [78, 180, 255], "isthing": 1, "id": 37, "name": "sports ball"}, {"color": [0, 228, 0], "isthing": 1, "id": 38, "name": "kite"}, {"color": [174, 255, 243], "isthing": 1, "id": 39, "name": "baseball bat"}, {"color": [45, 89, 255], "isthing": 1, "id": 40, "name": "baseball glove"}, {"color": [134, 134, 103], "isthing": 1, "id": 41, "name": "skateboard"}, {"color": [145, 148, 174], "isthing": 1, "id": 42, "name": "surfboard"}, {"color": [255, 208, 186], "isthing": 1, "id": 43, "name": "tennis racket"}, {"color": [197, 226, 255], "isthing": 1, "id": 44, "name": "bottle"}, {"color": [171, 134, 1], "isthing": 1, "id": 46, "name": "wine glass"}, {"color": [109, 63, 54], "isthing": 1, "id": 47, "name": "cup"}, {"color": [207, 138, 255], "isthing": 1, "id": 48, "name": "fork"}, {"color": [151, 0, 95], "isthing": 1, "id": 49, "name": "knife"}, {"color": [9, 80, 61], "isthing": 1, "id": 50, "name": "spoon"}, {"color": [84, 105, 51], "isthing": 1, "id": 51, "name": "bowl"}, {"color": [74, 65, 105], "isthing": 1, "id": 52, "name": "banana"}, {"color": [166, 196, 102], "isthing": 1, "id": 53, "name": "apple"}, {"color": [208, 195, 210], "isthing": 1, "id": 54, "name": "sandwich"}, {"color": [255, 109, 65], "isthing": 1, "id": 55, "name": "orange"}, {"color": [0, 143, 149], "isthing": 1, "id": 56, "name": "broccoli"}, {"color": [179, 0, 194], "isthing": 1, "id": 57, "name": "carrot"}, {"color": [209, 99, 106], "isthing": 1, "id": 58, "name": "hot dog"}, {"color": [5, 121, 0], "isthing": 1, "id": 59, "name": "pizza"}, {"color": [227, 255, 205], "isthing": 1, "id": 60, "name": "donut"}, {"color": [147, 186, 208], "isthing": 1, "id": 61, "name": "cake"}, {"color": [153, 69, 1], "isthing": 1, "id": 62, "name": "chair"}, {"color": [3, 95, 161], "isthing": 1, "id": 63, "name": "couch"}, {"color": [163, 255, 0], "isthing": 1, "id": 64, "name": "potted plant"}, {"color": [119, 0, 170], "isthing": 1, "id": 65, "name": "bed"}, {"color": [0, 182, 199], "isthing": 1, "id": 67, "name": "dining table"}, {"color": [0, 165, 120], "isthing": 1, "id": 70, "name": "toilet"}, {"color": [183, 130, 88], "isthing": 1, "id": 72, "name": "tv"}, {"color": [95, 32, 0], "isthing": 1, "id": 73, "name": "laptop"}, {"color": [130, 114, 135], "isthing": 1, "id": 74, "name": "mouse"}, {"color": [110, 129, 133], "isthing": 1, "id": 75, "name": "remote"}, {"color": [166, 74, 118], "isthing": 1, "id": 76, "name": "keyboard"}, {"color": [219, 142, 185], "isthing": 1, "id": 77, "name": "cell phone"}, {"color": [79, 210, 114], "isthing": 1, "id": 78, "name": "microwave"}, {"color": [178, 90, 62], "isthing": 1, "id": 79, "name": "oven"}, {"color": [65, 70, 15], "isthing": 1, "id": 80, "name": "toaster"}, {"color": [127, 167, 115], "isthing": 1, "id": 81, "name": "sink"}, {"color": [59, 105, 106], "isthing": 1, "id": 82, "name": "refrigerator"}, {"color": [142, 108, 45], "isthing": 1, "id": 84, "name": "book"}, {"color": [196, 172, 0], "isthing": 1, "id": 85, "name": "clock"}, {"color": [95, 54, 80], "isthing": 1, "id": 86, "name": "vase"}, {"color": [128, 76, 255], "isthing": 1, "id": 87, "name": "scissors"}, {"color": [201, 57, 1], "isthing": 1, "id": 88, "name": "teddy bear"}, {"color": [246, 0, 122], "isthing": 1, "id": 89, "name": "hair drier"}, {"color": [191, 162, 208], "isthing": 1, "id": 90, "name": "toothbrush"}, {"color": [255, 255, 128], "isthing": 0, "id": 92, "name": "banner"}, {"color": [147, 211, 203], "isthing": 0, "id": 93, "name": "blanket"}, {"color": [150, 100, 100], "isthing": 0, "id": 95, "name": "bridge"}, {"color": [168, 171, 172], "isthing": 0, "id": 100, "name": "cardboard"}, {"color": [146, 112, 198], "isthing": 0, "id": 107, "name": "counter"}, {"color": [210, 170, 100], "isthing": 0, "id": 109, "name": "curtain"}, {"color": [92, 136, 89], "isthing": 0, "id": 112, "name": "door"}, {"color": [218, 88, 184], "isthing": 0, "id": 118, "name": "wood floor"}, {"color": [241, 129, 0], "isthing": 0, "id": 119, "name": "flower"}, {"color": [217, 17, 255], "isthing": 0, "id": 122, "name": "fruit"}, {"color": [124, 74, 181], "isthing": 0, "id": 125, "name": "gravel"}, {"color": [70, 70, 70], "isthing": 0, "id": 128, "name": "house"}, {"color": [255, 228, 255], "isthing": 0, "id": 130, "name": "light"}, {"color": [154, 208, 0], "isthing": 0, "id": 133, "name": "mirror-stuff"}, {"color": [193, 0, 92], "isthing": 0, "id": 138, "name": "net"}, {"color": [76, 91, 113], "isthing": 0, "id": 141, "name": "pillow"}, {"color": [255, 180, 195], "isthing": 0, "id": 144, "name": "platform"}, {"color": [106, 154, 176], "isthing": 0, "id": 145, "name": "playingfield"}, {"color": [230, 150, 140], "isthing": 0, "id": 147, "name": "railroad"}, {"color": [60, 143, 255], "isthing": 0, "id": 148, "name": "river"}, {"color": [128, 64, 128], "isthing": 0, "id": 149, "name": "road"}, {"color": [92, 82, 55], "isthing": 0, "id": 151, "name": "roof"}, {"color": [254, 212, 124], "isthing": 0, "id": 154, "name": "sand"}, {"color": [73, 77, 174], "isthing": 0, "id": 155, "name": "sea"}, {"color": [255, 160, 98], "isthing": 0, "id": 156, "name": "shelf"}, {"color": [255, 255, 255], "isthing": 0, "id": 159, "name": "snow"}, {"color": [104, 84, 109], "isthing": 0, "id": 161, "name": "stairs"}, {"color": [169, 164, 131], "isthing": 0, "id": 166, "name": "tent"}, {"color": [225, 199, 255], "isthing": 0, "id": 168, "name": "towel"}, {"color": [137, 54, 74], "isthing": 0, "id": 171, "name": "brick wall"}, {"color": [135, 158, 223], "isthing": 0, "id": 175, "name": "stone wall"}, {"color": [7, 246, 231], "isthing": 0, "id": 176, "name": "tile wall"}, {"color": [107, 255, 200], "isthing": 0, "id": 177, "name": "wood wall"}, {"color": [58, 41, 149], "isthing": 0, "id": 178, "name": "water"}, {"color": [183, 121, 142], "isthing": 0, "id": 180, "name": "window blind"}, {"color": [255, 73, 97], "isthing": 0, "id": 181, "name": "window"}, {"color": [107, 142, 35], "isthing": 0, "id": 184, "name": "tree"}, {"color": [190, 153, 153], "isthing": 0, "id": 185, "name": "fence"}, {"color": [146, 139, 141], "isthing": 0, "id": 186, "name": "ceiling"}, {"color": [70, 130, 180], "isthing": 0, "id": 187, "name": "sky"}, {"color": [134, 199, 156], "isthing": 0, "id": 188, "name": "cabinet"}, {"color": [209, 226, 140], "isthing": 0, "id": 189, "name": "table"}, {"color": [96, 36, 108], "isthing": 0, "id": 190, "name": "floor"}, {"color": [96, 96, 96], "isthing": 0, "id": 191, "name": "pavement"}, {"color": [64, 170, 64], "isthing": 0, "id": 192, "name": "mountain"}, {"color": [152, 251, 152], "isthing": 0, "id": 193, "name": "grass"}, {"color": [208, 229, 228], "isthing": 0, "id": 194, "name": "dirt"}, {"color": [206, 186, 171], "isthing": 0, "id": 195, "name": "paper"}, {"color": [152, 161, 64], "isthing": 0, "id": 196, "name": "food"}, {"color": [116, 112, 0], "isthing": 0, "id": 197, "name": "building"}, {"color": [0, 114, 143], "isthing": 0, "id": 198, "name": "rock"}, {"color": [102, 102, 156], "isthing": 0, "id": 199, "name": "wall"}, {"color": [250, 141, 255], "isthing": 0, "id": 200, "name": "rug"}]

cityscapes_categories = [{"color": [128, 64, 128], "id": 7, "isthing": 0, "name": "road", "supercategory": "flat"},
                         {"color": [244, 35, 232], "id": 8, "isthing": 0, "name": "sidewalk", "supercategory": "flat"},
                         {"color": [70, 70, 70], "id": 11, "isthing": 0, "name": "building", "supercategory": "construction"},
                         {"color": [102, 102, 156], "id": 12, "isthing": 0, "name": "wall", "supercategory": "construction"},
                         {"color": [190, 153, 153], "id": 13, "isthing": 0, "name": "fence", "supercategory": "construction"},
                         {"color": [153, 153, 153], "id": 17, "isthing": 0, "name": "pole", "supercategory": "object"},
                         {"color": [250, 170, 30], "id": 19, "isthing": 0, "name": "traffic light", "supercategory": "object"},
                         {"color": [220, 220, 0], "id": 20, "isthing": 0, "name": "traffic sign", "supercategory": "object"},
                         {"color": [107, 142, 35], "id": 21, "isthing": 0, "name": "vegetation", "supercategory": "nature"},
                         {"color": [152, 251, 152], "id": 22, "isthing": 0, "name": "terrain", "supercategory": "nature"},
                         {"color": [70, 130, 180], "id": 23, "isthing": 0, "name": "sky", "supercategory": "sky"},
                         {"color": [220, 20, 60], "id": 24, "isthing": 1, "name": "person", "supercategory": "human"},
                         {"color": [255, 0, 0], "id": 25, "isthing": 1, "name": "rider", "supercategory": "human"},
                         {"color": [0, 0, 142], "id": 26, "isthing": 1, "name": "car", "supercategory": "vehicle"},
                         {"color": [0, 0, 70], "id": 27, "isthing": 1, "name": "truck", "supercategory": "vehicle"},
                         {"color": [0, 60, 100], "id": 28, "isthing": 1, "name": "bus", "supercategory": "vehicle"},
                         {"color": [0, 80, 100], "id": 31, "isthing": 1, "name": "train", "supercategory": "vehicle"},
                         {"color": [0, 0, 230], "id": 32, "isthing": 1, "name": "motorcycle", "supercategory": "vehicle"},
                         {"color": [119, 11, 32], "id": 33, "isthing": 1, "name": "bicycle", "supercategory": "vehicle"}]

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class MLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_layers: int,
            sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x

def show_anns(masks, input_point, input_box, input_label, filename, image, ious, boundary_ious):
    if len(masks) == 0:
        return

    for i, (mask, iou, biou) in enumerate(zip(masks, ious, boundary_ious)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        if input_box is not None:
            show_box(input_box, plt.gca())
        if (input_point is not None) and (input_label is not None):
            show_points(input_point, input_label, plt.gca())

        plt.axis('off')
        plt.savefig(filename + '_' + str(i) + '.png', bbox_inches='tight', pad_inches=-0.1)
        plt.close()


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def get_args_parser():
    parser = argparse.ArgumentParser('UnSeg', add_help=False)

    parser.add_argument("--checkpoint", type=str, default="./weight_dir/generator_weight_final_3e4.pth",
                        help="The path to the SAM checkpoint to use for mask generation.")

    parser.add_argument("--output", type=str, required=True,
                        help="Path to the directory where masks and checkpoints will be output")
    parser.add_argument("--model-type", type=str, default="vit_b",
                        help="The type of model to load, in ['vit_h', 'vit_l', 'vit_b']")

    parser.add_argument("--device", type=str, default="cuda",
                        help="The device to run generation on.")

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--poison_datasets', default='pascal', type=str)
    parser.add_argument('--infer_mode', default='single', type=str)
    parser.add_argument('--target_class', default=6.0, type=float)
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--epsilon', default=0.0627450980392157, type=float)
    parser.add_argument('--epsilon_bg', default=0.0156862745098039, type=float)

    parser.add_argument('--gpu_ids', default='0', type=str)
    parser.add_argument('--img_dir', default='./poison_all', type=str)
    parser.add_argument('--noise_dir', default='./noise_all', type=str)
    parser.add_argument('--source_img', default='./source_img', type=str)
    parser.add_argument('--source_gt', default='./source_gt', type=str)

    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--lr_drop_epoch', default=40, type=int)
    parser.add_argument('--max_epoch_num', default=48, type=int)
    parser.add_argument('--input_size', default=[1024, 1024], type=list)
    parser.add_argument('--batch_size_train', default=4, type=int)
    parser.add_argument('--batch_size_valid', default=1, type=int)
    parser.add_argument('--model_save_fre', default=1, type=int)

    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', type=int, help='local rank for dist')
    parser.add_argument('--find_unused_params', action='store_true')

    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--visualize', action='store_true', default=False)
    parser.add_argument("--restore-model", type=str,
                        help="The path to the hq_decoder training checkpoint for evaluation")

    return parser.parse_args()


def main(net, train_datasets, valid_datasets, args):
    misc.init_distributed_mode(args)
    print('world size: {}'.format(args.world_size))
    print('rank: {}'.format(args.rank))
    print('local_rank: {}'.format(args.local_rank))
    print("args: " + str(args) + '\n')

    ### --- Step 1: Poison dataset ---
    print("--- create valid dataloader ---")
    valid_im_gt_list = get_im_gt_name_dict(valid_datasets, flag="valid")
    valid_dataloaders, valid_datasets = create_dataloaders(valid_im_gt_list,
                                                           my_transforms=[
                                                               Resize_Padding(args.input_size, args.poison_datasets)
                                                           ],
                                                           batch_size=args.batch_size_valid,
                                                           training=False)
    print(len(valid_dataloaders), " valid dataloaders created")

    ### --- Step 2: DistributedDataParallel---
    if torch.cuda.is_available():
        net.cuda()
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu], find_unused_parameters=False)
    ### --- Step 3: Infer dataset ---
    # if torch.cuda.is_available():
    net.module.generator.load_state_dict(torch.load(args.checkpoint))
    print("weight has been loaded successfully...")
    evaluate(args, net, valid_dataloaders, args.visualize)




def compute_iou(preds, target):
    assert target.shape[1] == 1, 'only support one mask per image now'
    if (preds.shape[2] != target.shape[2] or preds.shape[3] != target.shape[3]):
        postprocess_preds = F.interpolate(preds, size=target.size()[2:], mode='bilinear', align_corners=False)
    else:
        postprocess_preds = preds
    iou = 0
    for i in range(0, len(preds)):
        iou = iou + misc.mask_iou(postprocess_preds[i], target[i])
    return iou / len(preds)


def compute_boundary_iou(preds, target):
    assert target.shape[1] == 1, 'only support one mask per image now'
    if (preds.shape[2] != target.shape[2] or preds.shape[3] != target.shape[3]):
        postprocess_preds = F.interpolate(preds, size=target.size()[2:], mode='bilinear', align_corners=False)
    else:
        postprocess_preds = preds
    iou = 0
    for i in range(0, len(preds)):
        iou = iou + misc.boundary_iou(target[i], postprocess_preds[i])
    return iou / len(preds)

#
def evaluate(args, net, valid_dataloaders, visualize=False):
    save_dir = args.img_dir
    save_dir_noise = args.noise_dir
    try:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    except FileExistsError:
        pass
    try:
        if not os.path.exists(save_dir_noise):
            os.makedirs(save_dir_noise)
    except FileExistsError:
        pass
    net.eval()
    print("Validating...")
    test_stats = {}
    index = 0

    max_value = 0
    min_value = 0
    for k in range(len(valid_dataloaders)):
        metric_logger = misc.MetricLogger(delimiter="  ")
        valid_dataloader = valid_dataloaders[k]
        print('valid_dataloader len:', len(valid_dataloader))

        for data_val in metric_logger.log_every(valid_dataloader, 100):
            imidx_val, inputs_val, labels_val, shapes_val, labels_ori = data_val['imidx'], data_val['image'], data_val[
                'label'], data_val['shape'], data_val['ori_label']
            img_name = data_val['name']
            ori_image = data_val['ori_image']
            pad_image = data_val['pad_image']
            padh = data_val['padh']
            padw = data_val['padw']
            gt_temp = data_val['gt_temp']
            if torch.cuda.is_available():
                inputs_val = inputs_val.cuda()
                labels_val = labels_val.cuda()
                gt_temp = gt_temp.cuda()

            imgs = inputs_val.permute(0, 2, 3, 1).cpu().numpy()
            label_mask = gt_temp / 255.0
            labels_256 = F.interpolate(label_mask, size=(256, 256), mode='bilinear')
            labels_256 = torch.round(labels_256) * 255.0
            labels_256_inverse =  (1.0 - labels_256 / 255.0) * 255.0
            reshaped_labels_256 = F.interpolate(labels_256 / 255.0, size=pad_image.shape[-2:], mode='bilinear')
            reshaped_labels_256 = torch.round(reshaped_labels_256)

            reshaped_labels_256_inverse = 1.0 - reshaped_labels_256

            input_keys = ['noise_mask']
            batched_input = []
            for b_i in range(len(imgs)):
                dict_input = dict()
                input_image = torch.as_tensor(imgs[b_i].astype(dtype=float), device=net.device, dtype=torch.float32).permute(2, 0,
                                                                                                             1).contiguous()
                dict_input['image'] = input_image
                dict_input['mask_inputs'] = labels_256[b_i:b_i + 1]
                dict_input['original_size'] = imgs[b_i].shape[:2]

                batched_input.append(dict_input)
            if torch.sum(label_mask) != 0:
                index = index + 1
                with torch.no_grad():
                    masks_hq_init = net(input=batched_input)
                masks_hq = F.interpolate(masks_hq_init, size=pad_image.shape[-2:], mode='bilinear')
                masks_hq = masks_hq * reshaped_labels_256 * args.epsilon + masks_hq * reshaped_labels_256_inverse * args.epsilon_bg
                masks_hq = masks_hq[:, :, 0:ori_image.shape[-2], 0:ori_image.shape[-1]]

                new_image = torch.clamp(ori_image + masks_hq.cpu() * 255.0, min=0, max=255).squeeze(0)
                new_image = torch.transpose(torch.transpose(new_image, 0, 1), 1, 2).numpy()[:, :, ::-1]
                save_path = os.path.join(save_dir, img_name[0]+'.jpg')
                cv2.imwrite(save_path, new_image)
            else:
                new_image = torch.transpose(torch.transpose(ori_image.squeeze(0), 0, 1), 1, 2).numpy()[:, :, ::-1]
                save_path = os.path.join(save_dir, img_name[0] + '.jpg')
                cv2.imwrite(save_path, new_image)

class BaseModel(nn.Module):
    def __init__(self,
                 generator: nn.Module,
                 ):
        super().__init__()
        self.generator = generator

    @property
    def device(self) -> Any:
        return self.generator.device

    def forward(self, input):
        noise = self.generator(input)
        return noise

if __name__ == "__main__":
    ### --------------- Configuring the Train and Valid datasets ---------------

    args = get_args_parser()

    _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    select_dataset = args.poison_datasets
    if select_dataset == 'pascal':
        dataset_voc_train = {"name": "PASCAL_VOC",
                       "im_dir": "./VOCtrainval_11-May-2012/pascal_aug/image_aug",
                       "gt_dir": "./VOCtrainval_11-May-2012/pascal_aug/label_aug",
                       "im_ext": ".jpg",
                       "gt_ext": ".png"}
    elif select_dataset == 'ade20k':
        ade20k_img_dir = os.path.join(_root, "ADEChallengeData2016/images/training_ori")
        ade20k_label_dir = os.path.join(_root, "ADEChallengeData2016/annotations_detectron2/training")
        dataset_voc_train = {"name": "ade20k",
                       "im_dir": ade20k_img_dir,
                       "gt_dir": ade20k_label_dir,
                       "im_ext": ".jpg",
                       "gt_ext": ".png"}
    elif select_dataset == 'pascal_tiny':
        dataset_voc_train = {"name": "PASCAL_VOC",
                       "im_dir": "./VOCtrainval_11-May-2012/train_set_1464",
                       "gt_dir": "./VOCtrainval_11-May-2012/VOCdevkit/VOC2012/SegmentationClass",
                       "im_ext": ".jpg",
                       "gt_ext": ".png"}
    elif select_dataset == 'coco':
        coco_img_dir = os.path.join(_root, "coco/train2017")
        coco_label_dir = os.path.join(_root, "coco/panoptic_train2017")
        dataset_voc_train = {"name": "coco",
                             "im_dir": coco_img_dir,
                             "gt_dir": coco_label_dir,
                             "im_ext": ".jpg",
                             "gt_ext": ".png"}
    elif select_dataset == 'cityscapes':
        cityscapes_img_dir = os.path.join(_root, "cityscapes/leftImg8bit/train")
        cityscapes_label_dir = os.path.join(_root, "cityscapes/gtFine/cityscapes_panoptic_train")
        dataset_voc_train = {"name": "cityscapes",
                             "im_dir": cityscapes_img_dir,
                             "gt_dir": cityscapes_label_dir,
                             "im_ext": ".png",
                             "gt_ext": ".png"}
    elif select_dataset == 'remote_sensing':
        remote_img_dir = '/remote-home/sunye/RSPrompter/data/SSDD/train_imgs'
        remote_label_dir = '/remote-home/sunye/RSPrompter/data/SSDD/annotations/train_instance_masks'
        dataset_voc_train = {"name": "remote_sensing",
                             "im_dir": remote_img_dir,
                             "gt_dir": remote_label_dir,
                             "im_ext": ".jpg",
                             "gt_ext": ".png"}
    elif select_dataset == 'interactive':
        remote_img_dir = args.source_img
        remote_label_dir = args.source_gt
        dataset_voc_train = {"name": "interactive",
                             "im_dir": remote_img_dir,
                             "gt_dir": remote_label_dir,
                             "im_ext": ".jpg",
                             "gt_ext": ".png"}
    else:
        print("wrong datasets...")
        exit(-1)
    infer_datasets = [dataset_voc_train]

    gpu_id = args.gpu_ids
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    net = BaseModel(
        generator=noise_sam_model_registry[args.model_type](checkpoint=None, epsilon=args.epsilon)
    )
    start = time.time()
    main(net, None, infer_datasets, args)
    end = time.time()
    cost = (end - start) / 3600
    payload = "Running Cost %.2f Hours \n" % cost
    print("payload = {}".format(payload))