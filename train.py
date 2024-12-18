# Copyright by UnSeg team
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
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

from utils.dataloader import get_im_gt_name_dict, create_dataloaders, RandomHFlip, Resize, LargeScaleJitter
from utils.loss_mask import loss_masks
import utils.misc as misc

from detectron2.data.common import DatasetFromList, MapDataset
from detectron2.data.samplers import TrainingSampler

os.environ["CUDA_VISIBLE_DEVICES"] =  os.getenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")

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

    parser.add_argument('--model_name', default='unseg', type=str)
    parser.add_argument("--output", type=str, required=True,
                        help="Path to the directory where masks and checkpoints will be output")
    parser.add_argument("--model-type", type=str, default="vit_b",
                        help="The type of model to load, in ['vit_h', 'vit_l', 'vit_b']")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="The path to the SAM checkpoint to use for mask generation.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="The device to run generation on.")

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--epsilon', default=0.003921568627451, type=float)
    parser.add_argument('--epsilon_bg', default=9.803921568627451e-4, type=float)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--lr_drop_epoch', default=20, type=int)
    parser.add_argument('--max_epoch_num', default=27, type=int)
    parser.add_argument('--input_size', default=[1024, 1024], type=list)
    parser.add_argument('--batch_size_train', default=4, type=int)
    parser.add_argument('--batch_size_valid', default=1, type=int)
    parser.add_argument('--surrogate_epoch', default=4, type=int)
    parser.add_argument('--generator_mode', default='label_modify', type=str)
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

def custom_collate(batch):
    elem_type = type(batch[0])
    if isinstance(batch[0], torch.Tensor):
        if not all(elem.size() == batch[0].size() for elem in batch):
            raise RuntimeError("All tensors in batch must have the same size")

        return torch.stack(batch, 0)
    elif isinstance(batch[0], dict):
        return {key: custom_collate([d[key] for d in batch]) for key in batch[0]}
    else:
        return default_collate(batch)

def main(net, train_datasets, valid_datasets, args):
    misc.init_distributed_mode(args)
    print('world size: {}'.format(args.world_size))
    print('rank: {}'.format(args.rank))
    print('local_rank: {}'.format(args.local_rank))
    print("args: " + str(args) + '\n')

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    ### --- Step 1: Train or Valid dataset ---
    if not args.eval:
        print("--- create training dataloader ---")
        train_im_gt_list = get_im_gt_name_dict(train_datasets, flag="train")
        train_dataloaders, train_datasets = create_dataloaders(train_im_gt_list,
                                                               my_transforms=[
                                                                   RandomHFlip(),
                                                                   LargeScaleJitter()
                                                               ],
                                                               batch_size=args.batch_size_train,
                                                               training=True)
        print(len(train_dataloaders), " train dataloaders created")
    print("--- create valid dataloader ---")
    valid_im_gt_list = get_im_gt_name_dict(valid_datasets, flag="valid")
    valid_dataloaders, valid_datasets = create_dataloaders(valid_im_gt_list,
                                                           my_transforms=[
                                                               Resize(args.input_size)
                                                           ],
                                                           batch_size=args.batch_size_valid,
                                                           training=False)
    print(len(valid_dataloaders), " valid dataloaders created")

    ### --- Step 2: DistributedDataParallel---
    if torch.cuda.is_available():
        net.cuda()
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu], find_unused_parameters=True)
    net_without_ddp = net.module
    ### --- Step 3: Train or Evaluate ---
    if not args.eval:
        print("--- define optimizer ---")
        optimizer = optim.Adam(net_without_ddp.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08,
                               weight_decay=0)

        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop_epoch)
        lr_scheduler.last_epoch = args.start_epoch
        train(args, net, optimizer, train_dataloaders, valid_dataloaders, lr_scheduler)
        torch.save(net.module.generator.state_dict(), './Generator_{}_final.pth'.format(args.model_name))
    else:
        if args.restore_model:
            print("restore model from:", args.restore_model)
            if torch.cuda.is_available():
                net.load_state_dict(torch.load(args.restore_model))
        evaluate(args, net, valid_dataloaders, args.visualize)

def train(args, net, optimizer, train_dataloaders, valid_dataloaders, lr_scheduler):
    if misc.is_main_process():
        os.makedirs(args.output, exist_ok=True)

    epoch_start = args.start_epoch
    epoch_num = args.max_epoch_num
    train_num = len(train_dataloaders)

    net.train()
    _ = net.to(device=args.device)
    for epoch in range(epoch_start, epoch_num):
        start = time.time()
        if epoch % args.surrogate_epoch == 0:
            for params in net.module.parameters():
                params.requires_grad = False
            for params in net.module.surrogate_model.parameters():
                params.requires_grad = True
        else:
            for name, params in net.module.named_parameters():
                if 'noise' in name:
                    params.requires_grad = True
                else:
                    params.requires_grad = False
        print("***Epoch:   ", epoch, "  learning rate:  ", optimizer.param_groups[0]["lr"], "***")
        metric_logger = misc.MetricLogger(delimiter="  ")
        train_dataloaders.batch_sampler.sampler.set_epoch(epoch)
        index = 0
        for data in metric_logger.log_every(train_dataloaders, 200):
            inputs, labels = data['image'], data['label']
            resize_inputs, resize_labels = data['resize_image'], data['resize_label']
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                resize_inputs = resize_inputs.cuda()
                labels = labels.cuda()
                resize_labels = resize_labels.cuda()

            imgs = inputs.permute(0, 2, 3, 1).cpu().numpy()
            resize_imgs = resize_inputs.permute(0, 2, 3, 1).cpu().numpy()
            input_keys = ['box', 'point', 'noise_mask']
            labels_box = misc.masks_to_boxes(labels[:, 0, :, :])
            labels_box_resize = misc.masks_to_boxes(resize_labels[:, 0, :, :])
            try:
                labels_points = misc.masks_sample_points(labels[:, 0, :, :])
                labels_points_resize = misc.masks_sample_points(resize_labels[:, 0, :, :])
            except:
                # less than 10 points
                input_keys = ['box', 'noise_mask']
            labels_256 = F.interpolate(labels, size=(256, 256), mode='bilinear')
            labels_noisemask = misc.masks_noise(labels_256)
            labels_128 = F.interpolate(resize_labels, size=(128, 128), mode='bilinear')
            labels_noisemask_resize = misc.masks_noise(labels_128)

            batched_input = []
            for b_i in range(len(imgs)):
                dict_input = dict()
                input_image = torch.as_tensor(imgs[b_i].astype(dtype=float), device=net.device, dtype=torch.float32).permute(2, 0,
                                                                                                           1).contiguous()
                resize_input_image = torch.as_tensor(resize_imgs[b_i].astype(dtype=float), device=net.device, dtype=torch.float32).permute(2, 0,
                                                                                                           1).contiguous()
                dict_input['image'] = input_image
                input_type_1 = random.choice(input_keys)

                num_elements = random.randint(1, len(input_keys))
                input_type = random.sample(input_keys, num_elements)

                if 'box' in input_type:
                    dict_input['boxes'] = labels_box[b_i:b_i + 1]
                if 'point' in input_type:
                    point_coords = labels_points[b_i:b_i + 1]
                    dict_input['point_coords'] = point_coords
                    dict_input['point_labels'] = torch.ones(point_coords.shape[1], device=point_coords.device)[None, :]
                if 'noise_mask' in input_type:
                    dict_input['mask_inputs'] = labels_noisemask[b_i:b_i + 1]
                # else:
                #     raise NotImplementedError
                dict_input['original_size'] = imgs[b_i].shape[:2]

                dict_input['original_label'] = labels[b_i:b_i + 1]
                """For surrogate model"""
                dict_input['resize_image'] = resize_input_image
                if input_type_1 == 'box':
                    dict_input['resize_boxes'] = labels_box_resize[b_i:b_i + 1]
                elif input_type_1 == 'point':
                    point_coords = labels_points_resize[b_i:b_i + 1]
                    dict_input['resize_point_coords'] = point_coords
                    dict_input['resize_point_labels'] = torch.ones(point_coords.shape[1], device=point_coords.device)[None, :]
                elif input_type_1 == 'noise_mask':
                    dict_input['resize_mask_inputs'] = labels_noisemask_resize[b_i:b_i + 1]
                else:
                    raise NotImplementedError
                dict_input['original_size'] = imgs[b_i].shape[:2]
                dict_input['original_resize'] = resize_imgs[b_i].shape[:2]

                batched_input.append(dict_input)

            if epoch % args.surrogate_epoch == 0:
                mask_pred = net(input=batched_input, epoch=epoch)
                loss_mask, loss_dice = loss_masks(mask_pred, resize_labels / 255.0, len(mask_pred))
                loss = loss_mask + loss_dice
                loss_dict = {"loss_mask": loss_mask, "loss_dice": loss_dice}
            else:
                mask_pred_all = net(input=batched_input, epoch=epoch)
                if args.generator_mode == 'label_modify':
                    rewrite_label = torch.ones_like(resize_labels).cuda()
                    loss_mask, loss_dice = loss_masks(mask_pred_all, rewrite_label, len(mask_pred_all))
                else:
                    loss_mask, loss_dice = loss_masks(mask_pred, resize_labels / 255.0, len(mask_pred_all))
                loss = loss_mask + loss_dice
                loss_dict = {"loss_mask": loss_mask, "loss_dice": loss_dice}

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = misc.reduce_dict(loss_dict)
            losses_reduced_scaled = sum(loss_dict_reduced.values())
            loss_value = losses_reduced_scaled.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            metric_logger.update(training_loss=loss_value, **loss_dict_reduced)

        print("Finished epoch:      ", epoch)
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}

        lr_scheduler.step()
        test_stats = evaluate(args, net, valid_dataloaders, epoch)
        train_stats.update(test_stats)

        net.train()

        # if epoch % args.model_save_fre == 0:
        #     model_name = "/epoch_" + str(epoch) + ".pth"
        #     print('come here save at', args.output + model_name)
        #     misc.save_on_master(net.module.state_dict(), args.output + model_name)
        #     torch.save(net.module.generator.state_dict(), './Generator_epoch_{}_adv_4_1_ue10.pth'.format(epoch))

        end = time.time()
        cost = (end - start) / 3600
        payload = "Epoch running Cost %.2f Hours \n" % cost
        print("Epoch payload = {}".format(payload))

    # Finish training
    print("Training Reaches The Maximum Epoch Number")

    # merge sam and hq_decoder
    # if misc.is_main_process():
    #     sam_ckpt = torch.load(args.checkpoint)
    #     hq_decoder = torch.load(args.output + model_name)
    #     for key in hq_decoder.keys():
    #         sam_key = 'mask_decoder.' + key
    #         if sam_key not in sam_ckpt.keys():
    #             sam_ckpt[sam_key] = hq_decoder[key]
    #     model_name = "/sam_hq_epoch_" + str(epoch) + ".pth"
    #     torch.save(sam_ckpt, args.output + model_name)


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


def evaluate(args, net, valid_dataloaders, visualize=False, epoch=-1):
    net.eval()
    print("Validating...")
    test_stats = {}

    for k in range(len(valid_dataloaders)):
        metric_logger = misc.MetricLogger(delimiter="  ")
        valid_dataloader = valid_dataloaders[k]
        print('valid_dataloader len:', len(valid_dataloader))

        for data_val in metric_logger.log_every(valid_dataloader, 20):
            imidx_val, inputs_val, labels_val, shapes_val, labels_ori = data_val['imidx'], data_val['image'], data_val[
                'label'], data_val['shape'], data_val['ori_label']
            resize_img = data_val['resize_image']
            resize_label = data_val['resize_label']
            if torch.cuda.is_available():
                inputs_val = inputs_val.cuda()
                labels_val = labels_val.cuda()
                labels_ori = labels_ori.cuda()
                resize_img = resize_img.cuda()
                resize_label = resize_label.cuda()

            imgs = inputs_val.permute(0, 2, 3, 1).cpu().numpy()
            resize_imgs = resize_img.permute(0, 2, 3, 1).cpu().numpy()

            labels_box = misc.masks_to_boxes(labels_val[:, 0, :, :])
            labels_box_resize = misc.masks_to_boxes(resize_label[:, 0, :, :])
            input_keys = ['box']
            batched_input = []

            labels_256 = F.interpolate(labels_val, size=(256, 256), mode='bilinear')
            labels_noisemask = misc.masks_noise(labels_256)
            labels_128 = F.interpolate(resize_label, size=(128, 128), mode='bilinear')
            labels_noisemask_resize = misc.masks_noise(labels_128)

            for b_i in range(len(imgs)):
                dict_input = dict()
                input_image = torch.as_tensor(imgs[b_i].astype(dtype=float), device=net.device, dtype=torch.float32).permute(2, 0,
                                                                                                           1).contiguous()
                resize_input_image = torch.as_tensor(resize_imgs[b_i].astype(dtype=float), device=net.device, dtype=torch.float32).permute(2, 0,
                                                                                                           1).contiguous()
                dict_input['image'] = input_image
                input_type = random.choice(input_keys)
                if input_type == 'box':
                    dict_input['boxes'] = labels_box[b_i:b_i + 1]
                elif input_type == 'point':
                    point_coords = labels_points[b_i:b_i + 1]
                    dict_input['point_coords'] = point_coords
                    dict_input['point_labels'] = torch.ones(point_coords.shape[1], device=point_coords.device)[None, :]
                elif input_type == 'noise_mask':
                    dict_input['mask_inputs'] = labels_noisemask[b_i:b_i + 1]
                    dict_input['resize_mask_inputs'] = labels_noisemask_resize[b_i:b_i + 1]
                else:
                    raise NotImplementedError
                dict_input['original_size'] = imgs[b_i].shape[:2]
                dict_input['original_label'] = labels_val[b_i:b_i + 1]

                """For surrogate model"""
                dict_input['resize_image'] = resize_input_image
                if input_type == 'box':
                    dict_input['resize_boxes'] = labels_box_resize[b_i:b_i + 1]
                # elif input_type == 'point':
                #     point_coords = labels_points_resize[b_i:b_i + 1]
                #     dict_input['resize_point_coords'] = point_coords
                #     dict_input['resize_point_labels'] = torch.ones(point_coords.shape[1], device=point_coords.device)[
                #                                         None, :]
                # elif input_type == 'noise_mask':
                #     dict_input['resize_mask_inputs'] = labels_noisemask_resize[b_i:b_i + 1]
                # else:
                #     raise NotImplementedError
                # dict_input['original_size'] = imgs[b_i].shape[:2]
                dict_input['original_resize'] = resize_imgs[b_i].shape[:2]

                batched_input.append(dict_input)

            with torch.no_grad():
                masks_hq = net(batched_input, epoch=0)

            iou = compute_iou(masks_hq, labels_ori)
            boundary_iou = compute_boundary_iou(masks_hq, labels_ori)

            # if visualize:
            #     print("visualize")
            #     os.makedirs(args.output, exist_ok=True)
            #     masks_hq_vis = (F.interpolate(masks_hq.detach(), (1024, 1024), mode="bilinear",
            #                                   align_corners=False) > 0).cpu()
            #     for ii in range(len(imgs)):
            #         base = data_val['imidx'][ii].item()
            #         print('base:', base)
            #         save_base = os.path.join(args.output, str(k) + '_' + str(base))
            #         imgs_ii = imgs[ii].astype(dtype=np.uint8)
            #         show_iou = torch.tensor([iou.item()])
            #         show_boundary_iou = torch.tensor([boundary_iou.item()])
            #         show_anns(masks_hq_vis[ii], None, labels_box[ii].cpu(), None, save_base, imgs_ii, show_iou,
            #                   show_boundary_iou)

            loss_dict = {"val_iou_" + str(k): iou, "val_boundary_iou_" + str(k): boundary_iou}
            loss_dict_reduced = misc.reduce_dict(loss_dict)
            metric_logger.update(**loss_dict_reduced)

        print('============================')
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
        test_stats.update(resstat)

    return test_stats


class BaseModel(nn.Module):
    def __init__(self,
                 generator: nn.Module,
                 surrogate_model: nn.Module,
                 surrogate_epoch: int,
                 ):
        super().__init__()
        self.generator = generator
        self.surrogate_model = surrogate_model
        self.global_noise = None
        self.index = 0
        self.surrogate_epoch = surrogate_epoch

    def freeze_generator(self):
        for param in self.generator.parameters():
            param.requires_grad = False

    def freeze_surrogate(self):
        for param in self.surrogate_model.parameters():
            param.requires_grad = False

    def unfreeze_generator(self):
        for name, params in net.named_parameters():
            if 'noise' in name:
                params.requires_grad = True
            else:
                params.requires_grad = False

    def unfreeze_surrogate(self):
        for param in self.surrogate_model.parameters():
            param.requires_grad = True

    def get_generator_params(self):
        for name, param in self.named_parameters():
            if 'noise' in name:
                yield param

    @property
    def device(self) -> Any:
        return self.generator.device

    def forward(self, input, epoch=-1):
        noise = self.generator(input)
        mask_pred = self.surrogate_model(input, noise, epoch, noise_type='all')
        return mask_pred

if __name__ == "__main__":
    ### --------------- Configuring the Train and Valid datasets ---------------

    dataset_dis = {"name": "DIS5K-TR",
                   "im_dir": "./data/DIS5K/DIS-TR/im",
                   "gt_dir": "./data/DIS5K/DIS-TR/gt",
                   "im_ext": ".jpg",
                   "gt_ext": ".png"}

    dataset_thin = {"name": "ThinObject5k-TR",
                    "im_dir": "./data/thin_object_detection/ThinObject5K/images_train",
                    "gt_dir": "./data/thin_object_detection/ThinObject5K/masks_train",
                    "im_ext": ".jpg",
                    "gt_ext": ".png"}

    dataset_fss = {"name": "FSS",
                   "im_dir": "./data/cascade_psp/fss_all",
                   "gt_dir": "./data/cascade_psp/fss_all",
                   "im_ext": ".jpg",
                   "gt_ext": ".png"}

    dataset_duts = {"name": "DUTS-TR",
                    "im_dir": "./data/cascade_psp/DUTS-TR",
                    "gt_dir": "./data/cascade_psp/DUTS-TR",
                    "im_ext": ".jpg",
                    "gt_ext": ".png"}

    dataset_duts_te = {"name": "DUTS-TE",
                       "im_dir": "./data/cascade_psp/DUTS-TE",
                       "gt_dir": "./data/cascade_psp/DUTS-TE",
                       "im_ext": ".jpg",
                       "gt_ext": ".png"}

    dataset_ecssd = {"name": "ECSSD",
                     "im_dir": "./data/cascade_psp/ecssd",
                     "gt_dir": "./data/cascade_psp/ecssd",
                     "im_ext": ".jpg",
                     "gt_ext": ".png"}

    dataset_msra = {"name": "MSRA10K",
                    "im_dir": "./data/cascade_psp/MSRA_10K",
                    "gt_dir": "./data/cascade_psp/MSRA_10K",
                    "im_ext": ".jpg",
                    "gt_ext": ".png"}

    # valid set
    dataset_coift_val = {"name": "COIFT",
                         "im_dir": "./data/thin_object_detection/COIFT/images",
                         "gt_dir": "./data/thin_object_detection/COIFT/masks",
                         "im_ext": ".jpg",
                         "gt_ext": ".png"}

    dataset_hrsod_val = {"name": "HRSOD",
                         "im_dir": "./data/thin_object_detection/HRSOD/images",
                         "gt_dir": "./data/thin_object_detection/HRSOD/masks_max255",
                         "im_ext": ".jpg",
                         "gt_ext": ".png"}

    dataset_thin_val = {"name": "ThinObject5k-TE",
                        "im_dir": "./data/thin_object_detection/ThinObject5K/images_test",
                        "gt_dir": "./data/thin_object_detection/ThinObject5K/masks_test",
                        "im_ext": ".jpg",
                        "gt_ext": ".png"}

    dataset_dis_val = {"name": "DIS5K-VD",
                       "im_dir": "./data/DIS5K/DIS-VD/im",
                       "gt_dir": "./data/DIS5K/DIS-VD/gt",
                       "im_ext": ".jpg",
                       "gt_ext": ".png"}

    args = get_args_parser()
    train_datasets = [dataset_dis, dataset_thin, dataset_fss, dataset_duts, dataset_duts_te, dataset_ecssd,
                      dataset_msra]
    valid_datasets = [dataset_dis_val, dataset_coift_val, dataset_hrsod_val, dataset_thin_val]

    net = BaseModel(
        generator=noise_sam_model_registry[args.model_type](checkpoint=None, epsilon=args.epsilon),
        surrogate_model=scratch_sam_model_registry[args.model_type](checkpoint=None, epsilon=args.epsilon, epsilon_bg=args.epsilon_bg),
        surrogate_epoch=args.surrogate_epoch,
    )
    start = time.time()
    main(net, train_datasets, valid_datasets, args)
    end = time.time()
    cost = (end - start) / 3600
    payload = "Running Cost %.2f Hours \n" % cost
    print("payload = {}".format(payload))
