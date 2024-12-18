# Copyright by UnSeg team
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

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
import random
from typing import Dict, List, Tuple, Any

from segment_anything_training import sam_model_registry, noise_sam_model_registry, scratch_sam_model_registry
from segment_anything_training.modeling import TwoWayTransformer, MaskDecoder

from utils.dataloader import get_im_gt_name_dict, create_dataloaders, RandomHFlip, Resize, LargeScaleJitter, Resize_Padding
from utils.loss_mask import loss_masks
import utils.misc as misc
from torchvision.transforms import ToPILImage

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

ADE_CATEGORIES = [{'id': 7, 'name': 'bed'}, {'id': 8, 'name': 'windowpane'}, {'id': 10, 'name': 'cabinet'}, {'id': 12, 'name': 'person'}, {'id': 14, 'name': 'door'}, {'id': 15, 'name': 'table'}, {'id': 18, 'name': 'curtain'}, {'id': 19, 'name': 'chair'}, {'id': 20, 'name': 'car'}, {'id': 22, 'name': 'painting'}, {'id': 23, 'name': 'sofa'}, {'id': 24, 'name': 'shelf'}, {'id': 27, 'name': 'mirror'}, {'id': 30, 'name': 'armchair'}, {'id': 31, 'name': 'seat'}, {'id': 32, 'name': 'fence'}, {'id': 33, 'name': 'desk'}, {'id': 35, 'name': 'wardrobe'}, {'id': 36, 'name': 'lamp'}, {'id': 37, 'name': 'bathtub'}, {'id': 38, 'name': 'railing'}, {'id': 39, 'name': 'cushion'}, {'id': 41, 'name': 'box'}, {'id': 42, 'name': 'column'}, {'id': 43, 'name': 'signboard'}, {'id': 44, 'name': 'chest of drawers'}, {'id': 45, 'name': 'counter'}, {'id': 47, 'name': 'sink'}, {'id': 49, 'name': 'fireplace'}, {'id': 50, 'name': 'refrigerator'}, {'id': 53, 'name': 'stairs'}, {'id': 55, 'name': 'case'}, {'id': 56, 'name': 'pool table'}, {'id': 57, 'name': 'pillow'}, {'id': 58, 'name': 'screen door'}, {'id': 62, 'name': 'bookcase'}, {'id': 64, 'name': 'coffee table'}, {'id': 65, 'name': 'toilet'}, {'id': 66, 'name': 'flower'}, {'id': 67, 'name': 'book'}, {'id': 69, 'name': 'bench'}, {'id': 70, 'name': 'countertop'}, {'id': 71, 'name': 'stove'}, {'id': 72, 'name': 'palm'}, {'id': 73, 'name': 'kitchen island'}, {'id': 74, 'name': 'computer'}, {'id': 75, 'name': 'swivel chair'}, {'id': 76, 'name': 'boat'}, {'id': 78, 'name': 'arcade machine'}, {'id': 80, 'name': 'bus'}, {'id': 81, 'name': 'towel'}, {'id': 82, 'name': 'light'}, {'id': 83, 'name': 'truck'}, {'id': 85, 'name': 'chandelier'}, {'id': 86, 'name': 'awning'}, {'id': 87, 'name': 'streetlight'}, {'id': 88, 'name': 'booth'}, {'id': 89, 'name': 'television receiver'}, {'id': 90, 'name': 'airplane'}, {'id': 92, 'name': 'apparel'}, {'id': 93, 'name': 'pole'}, {'id': 95, 'name': 'bannister'}, {'id': 97, 'name': 'ottoman'}, {'id': 98, 'name': 'bottle'}, {'id': 102, 'name': 'van'}, {'id': 103, 'name': 'ship'}, {'id': 104, 'name': 'fountain'}, {'id': 107, 'name': 'washer'}, {'id': 108, 'name': 'plaything'}, {'id': 110, 'name': 'stool'}, {'id': 111, 'name': 'barrel'}, {'id': 112, 'name': 'basket'}, {'id': 115, 'name': 'bag'}, {'id': 116, 'name': 'minibike'}, {'id': 118, 'name': 'oven'}, {'id': 119, 'name': 'ball'}, {'id': 120, 'name': 'food'}, {'id': 121, 'name': 'step'}, {'id': 123, 'name': 'trade name'}, {'id': 124, 'name': 'microwave'}, {'id': 125, 'name': 'pot'}, {'id': 126, 'name': 'animal'}, {'id': 127, 'name': 'bicycle'}, {'id': 129, 'name': 'dishwasher'}, {'id': 130, 'name': 'screen'}, {'id': 132, 'name': 'sculpture'}, {'id': 133, 'name': 'hood'}, {'id': 134, 'name': 'sconce'}, {'id': 135, 'name': 'vase'}, {'id': 136, 'name': 'traffic light'}, {'id': 137, 'name': 'tray'}, {'id': 138, 'name': 'ashcan'}, {'id': 139, 'name': 'fan'}, {'id': 142, 'name': 'plate'}, {'id': 143, 'name': 'monitor'}, {'id': 144, 'name': 'bulletin board'}, {'id': 146, 'name': 'radiator'}, {'id': 147, 'name': 'glass'}, {'id': 148, 'name': 'clock'}, {'id': 149, 'name': 'flag'}]

name_list = [item['id'] for item in ADE_CATEGORIES]

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

    parser.add_argument("--noise_type", type=str, default="min")
    parser.add_argument("--device", type=str, default="cuda",
                        help="The device to run generation on.")

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--poison_datasets', default='ade20k', type=str)
    parser.add_argument('--infer_mode', default='single', type=str)
    parser.add_argument('--target_class', default=6.0, type=float)
    parser.add_argument('--target_list', nargs='+', type=int, help='List of target classes')
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--epsilon', default=0.0627450980392157, type=float)
    parser.add_argument('--epsilon_bg', default=0.0156862745098039, type=float)

    parser.add_argument('--gpu_ids', default='0', type=str)
    parser.add_argument('--img_dir', default='./poison_all', type=str)
    parser.add_argument('--noise_dir', default='./noise_all', type=str)

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
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu],
                                                    find_unused_parameters=False)
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

# #
def evaluate(args, net, valid_dataloaders, visualize=False):
    save_dir = args.img_dir
    save_dir_noise = args.noise_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(save_dir_noise):
        os.makedirs(save_dir_noise)
    net.eval()
    print("Validating...")
    test_stats = {}
    index = 0

    # from PIL import Image
    # def label_to_color_image(label):
    #     color_image = np.zeros((1024, 1024, 3), dtype=np.uint8)
    #     cmap = plt.get_cmap('tab20b', 150)
    #     colors = cmap(np.linspace(0, 1, 150))[:, :3]
    #     for cls in range(1, 151):
    #         color = (colors[cls - 1] * 255).astype(np.uint8)
    #         mask = label == cls
    #         color_image[mask[0, 0]] = color  # Changed mask.squeeze(0) to mask[0, 0]
    #     return color_image

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
            label_mask = None
            if args.infer_mode == 'single':
                label_mask = (gt_temp == args.target_class).float()
            elif args.infer_mode == 'multi':
                # color_image = label_to_color_image(gt_temp.cpu().numpy())
                # if color_image.ndim == 4 and (color_image.shape[0] == 1 or color_image.shape[1] == 1):
                #     color_image = color_image.squeeze()
                # if color_image.dtype != np.uint8:
                #     color_image = (color_image * 255).astype(np.uint8)
                # img = Image.fromarray(color_image)
                # temp_folder = 'temp_folder'
                # save_path = os.path.join(temp_folder, img_name[0] + '_label.jpg')
                # save_path_img = os.path.join(temp_folder, img_name[0] + '_img.jpg')
                # if not os.path.exists(temp_folder):
                #     os.makedirs(temp_folder)
                # img.save(save_path)
                # if imgs.ndim == 4 and (imgs.shape[0] == 1 or imgs.shape[1] == 1):
                #     imgs = imgs.squeeze()
                # if imgs.dtype != np.uint8:
                #     imgs = (imgs * 255).astype(np.uint8)
                #
                # img = Image.fromarray(imgs)
                # imgs = Image.fromarray(imgs)
                #
                # imgs.save(save_path_img)
                # continue
                label_mask = torch.zeros_like(gt_temp).cuda()
                target_list = name_list
                for cls in target_list:
                    label_mask[gt_temp == cls] = 1.0
                label_mask = label_mask.float()
            elif args.infer_mode == 'all':
                label_mask = (gt_temp != 255).float()
            labels_256 = F.interpolate(label_mask, size=(256, 256), mode='bilinear')
            labels_256 = torch.round(labels_256) * 255.0
            reshaped_labels_256 = F.interpolate(labels_256 / 255.0, size=pad_image.shape[-2:], mode='bilinear')
            reshaped_labels_256 = torch.round(reshaped_labels_256)

            reshaped_labels_256_inverse = 1.0 - reshaped_labels_256

            labels_box = misc.masks_to_boxes(label_mask[:, 0, :, :])
            labels_points = misc.masks_sample_points(label_mask[:, 0, :, :], k=32)
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
                       "im_dir": "../VOCtrainval_11-May-2012/pascal_aug/image_aug",
                       "gt_dir": "../VOCtrainval_11-May-2012/pascal_aug/label_aug",
                       "im_ext": ".jpg",
                       "gt_ext": ".png"}
    elif select_dataset == 'ade20k':
        ade20k_img_dir = os.path.join(_root, "ADEChallengeData2016/images/training")
        ade20k_label_dir = os.path.join(_root, "ADEChallengeData2016/annotations_detectron2/training")
        dataset_voc_train = {"name": "ade20k",
                       "im_dir": ade20k_img_dir,
                       "gt_dir": ade20k_label_dir,
                       "im_ext": ".jpg",
                       "gt_ext": ".png"}
    elif select_dataset == 'pascal_tiny':
        dataset_voc_train = {"name": "PASCAL_VOC",
                       "im_dir": "../VOCtrainval_11-May-2012/train_set_1464",
                       "gt_dir": "../VOCtrainval_11-May-2012/VOCdevkit/VOC2012/SegmentationClass",
                       "im_ext": ".jpg",
                       "gt_ext": ".png"}
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