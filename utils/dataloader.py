# Copyright by UnSeg team
# All rights reserved.

## data loader
from __future__ import print_function, division
import cv2
import PIL.Image as Image
import numpy as np
import random
from copy import deepcopy
from skimage import io
import os
from glob import glob
from detectron2.data import detection_utils as detectron_utils
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms, utils
from torchvision.transforms.functional import normalize
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler

#### --------------------- dataloader online ---------------------####

class CocoPanopticDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.transform = transforms.Compose([
            RandomHFlip(),
            LargeScaleJitter()
        ])

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, idx):
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)


        img_path = self.datasets[idx]['file_name']
        pan_seg_path = self.datasets[idx]['pan_seg_file_name']
        image = Image.open(img_path).convert('RGB')
        pan_seg = Image.open(pan_seg_path).convert('L')  # 转换为灰度图
        image = self.transform(image)
        pan_seg = transforms.ToTensor()(pan_seg)
        return image, pan_seg

def get_im_gt_name_dict(datasets, flag='valid'):
    print("------------------------------", flag, "--------------------------------")
    name_im_gt_list = []

    for i in range(len(datasets)):
        print("--->>>", flag, " dataset ",i,"/",len(datasets)," ",datasets[i]["name"],"<<<---")
        tmp_im_list, tmp_gt_list = [], []
        tmp_im_list = glob(datasets[i]["im_dir"]+os.sep+'*'+datasets[i]["im_ext"])
        print('-im-',datasets[i]["name"],datasets[i]["im_dir"], ': ',len(tmp_im_list))

        if(datasets[i]["gt_dir"]==""):
            print('-gt-', datasets[i]["name"], datasets[i]["gt_dir"], ': ', 'No Ground Truth Found')
            tmp_gt_list = []
        else:
            tmp_gt_list = [datasets[i]["gt_dir"]+os.sep+x.split(os.sep)[-1].split(datasets[i]["im_ext"])[0]+datasets[i]["gt_ext"] for x in tmp_im_list]
            print('-gt-', datasets[i]["name"],datasets[i]["gt_dir"], ': ',len(tmp_gt_list))
        name_im_gt_list.append({"dataset_name":datasets[i]["name"],
                                "im_path":tmp_im_list,
                                "gt_path":tmp_gt_list,
                                "im_ext":datasets[i]["im_ext"],
                                "gt_ext":datasets[i]["gt_ext"]})
    return name_im_gt_list

def get_im_gt_name_dict(datasets, flag='valid'):
    print("------------------------------", flag, "--------------------------------")
    name_im_gt_list = []

    for i in range(len(datasets)):
        print("--->>>", flag, " dataset ",i,"/",len(datasets)," ",datasets[i]["name"],"<<<---")
        tmp_im_list, tmp_gt_list = [], []
        tmp_im_list = glob(datasets[i]["im_dir"]+os.sep+'*'+datasets[i]["im_ext"])
        print('-im-',datasets[i]["name"],datasets[i]["im_dir"], ': ',len(tmp_im_list))

        if(datasets[i]["gt_dir"]==""):
            print('-gt-', datasets[i]["name"], datasets[i]["gt_dir"], ': ', 'No Ground Truth Found')
            tmp_gt_list = []
        else:
            tmp_gt_list = [datasets[i]["gt_dir"]+os.sep+x.split(os.sep)[-1].split(datasets[i]["im_ext"])[0]+datasets[i]["gt_ext"] for x in tmp_im_list]
            print('-gt-', datasets[i]["name"],datasets[i]["gt_dir"], ': ',len(tmp_gt_list))
        name_im_gt_list.append({"dataset_name":datasets[i]["name"],
                                "im_path":tmp_im_list,
                                "gt_path":tmp_gt_list,
                                "im_ext":datasets[i]["im_ext"],
                                "gt_ext":datasets[i]["gt_ext"]})
    return name_im_gt_list


def get_im_gt_name_dict_cityscapes(datasets, flag='valid'):
    print("------------------------------", flag, "--------------------------------")
    name_im_gt_list = []

    for i in range(len(datasets)):
        print("--->>>", flag, " dataset ",i,"/",len(datasets)," ",datasets[i]["name"],"<<<---")
        tmp_im_list, tmp_gt_list = [], []
        tmp_im_list = glob(os.path.join(datasets[i]["im_dir"], '**', '*' + datasets[i]["im_ext"]), recursive=True) ###
        print('-im-',datasets[i]["name"],datasets[i]["im_dir"], ': ',len(tmp_im_list))
        if(datasets[i]["gt_dir"]==""):
            print('-gt-', datasets[i]["name"], datasets[i]["gt_dir"], ': ', 'No Ground Truth Found')
            tmp_gt_list = []
        else:
            tmp_gt_list = [datasets[i]["gt_dir"]+os.sep+x.split(os.sep)[-1].split(datasets[i]["im_ext"])[0]+datasets[i]["gt_ext"] for x in tmp_im_list]
            tmp_gt_list = [item.replace("leftImg8bit", "gtFine_panoptic") for item in tmp_gt_list]  ###
            print('-gt-', datasets[i]["name"],datasets[i]["gt_dir"], ': ',len(tmp_gt_list))
        name_im_gt_list.append({"dataset_name":datasets[i]["name"],
                                "im_path":tmp_im_list,
                                "gt_path":tmp_gt_list,
                                "im_ext":datasets[i]["im_ext"],
                                "gt_ext":datasets[i]["gt_ext"]})
    return name_im_gt_list


def create_dataloaders(name_im_gt_list, my_transforms=[], batch_size=1, training=False):
    gos_dataloaders = []
    gos_datasets = []

    if(len(name_im_gt_list)==0):
        return gos_dataloaders, gos_datasets

    num_workers_ = 1
    if(batch_size>1):
        num_workers_ = 2
    if(batch_size>4):
        num_workers_ = 4
    if(batch_size>8):
        num_workers_ = 8


    if training:
        for i in range(len(name_im_gt_list)):
            gos_dataset = OnlineDataset([name_im_gt_list[i]], transform = transforms.Compose(my_transforms))
            gos_datasets.append(gos_dataset)

        gos_dataset = ConcatDataset(gos_datasets)
        sampler = DistributedSampler(gos_dataset)
        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler, batch_size, drop_last=True)
        dataloader = DataLoader(gos_dataset, batch_sampler=batch_sampler_train, num_workers=num_workers_)

        gos_dataloaders = dataloader
        gos_datasets = gos_dataset

    else:
        for i in range(len(name_im_gt_list)):
            gos_dataset = OnlineDataset([name_im_gt_list[i]], transform = transforms.Compose(my_transforms), eval_ori_resolution = True)
            sampler = DistributedSampler(gos_dataset, shuffle=False)
            dataloader = DataLoader(gos_dataset, batch_size, sampler=sampler, drop_last=False, num_workers=num_workers_)

            gos_dataloaders.append(dataloader)
            gos_datasets.append(gos_dataset)

    return gos_dataloaders, gos_datasets

class RandomHFlip(object):
    def __init__(self,prob=0.5):
        self.prob = prob
    def __call__(self,sample):
        imidx, image, label, shape = sample['imidx'], sample['image'], sample['label'], sample['shape']
        # random horizontal flip
        if random.random() >= self.prob:
            image = torch.flip(image, dims=[2])
            label = torch.flip(label, dims=[2])
        if "classes" in sample.keys():
            return {'imidx':imidx,'image':image, 'label':label, 'shape':shape, 'name':sample['name'], 'classes':sample['classes']}
        return {'imidx':imidx,'image':image, 'label':label, 'shape':shape, 'name':sample['name']}

class Resize(object):
    def __init__(self,size=[1024,1024]):
        self.size = size
    def __call__(self,sample):
        imidx, image, label, shape = sample['imidx'], sample['image'], sample['label'], sample['shape']

        resize_image = torch.squeeze(F.interpolate(torch.unsqueeze(image, 0), [512, 512],mode='bilinear'),dim=0) / 255.0
        resize_label = torch.squeeze(F.interpolate(torch.unsqueeze(label, 0), [512, 512],mode='bilinear'),dim=0)

        image = torch.squeeze(F.interpolate(torch.unsqueeze(image, 0), self.size,mode='bilinear'),dim=0)/ 255.0
        label = torch.squeeze(F.interpolate(torch.unsqueeze(label, 0), self.size,mode='bilinear'),dim=0)

        return {'imidx':imidx,'image':image, 'name':sample['name'], 'label':label,'resize_image':resize_image,
                'resize_label':resize_label,'shape':torch.tensor(self.size), "ori_shape":shape}

class Resize_Padding(object):
    def __init__(self,size=[1024, 1024], datasets='pascal'):
        self.size = size
        self.img_size = 1024
        self.datasets = datasets
    def __call__(self,sample):
        imidx, image, label, shape = sample['imidx'], sample['image'], sample['label'], sample['shape']
        gt_path = sample['gt_path']
        image = torch.unsqueeze(image, 0)
        label = torch.unsqueeze(label, 0)
        gt_temp = None
        if self.datasets == 'pascal' or self.datasets == 'pascal_tiny' or self.datasets == 'cityscapes_local':
            gt_temp = np.asarray(Image.open(gt_path))
        elif self.datasets == 'ade20k':
            gt_temp = np.asarray(detectron_utils.read_image(gt_path))
        elif self.datasets == 'coco' or self.datasets == 'cityscapes':
            gt_temp = np.asarray(detectron_utils.read_image(gt_path, "RGB"))
            from panopticapi.utils import rgb2id
            gt_temp = rgb2id(gt_temp)
        elif self.datasets == 'remote_sensing':
            gt_temp = io.imread(gt_path)
        elif self.datasets == 'interactive' or self.datasets == 'saliency' or self.datasets == 'medical':
            gt_temp = io.imread(gt_path)
            if len(gt_temp.shape) > 2:
                gt_temp = gt_temp[:, :, 0]
        gt_temp = torch.from_numpy(gt_temp.copy())
        gt_temp = gt_temp.unsqueeze(0).unsqueeze(0)
        initial_gt = gt_temp.squeeze(0)
        h, w = shape
        max_size = max(h, w)
        padh = max_size - h
        padw = max_size - w
        pad_image = F.pad(image, (0, padw, 0, padh))
        pad_label = F.pad(label, (0, padw, 0, padh))
        gt_temp = F.pad(gt_temp, (0, padw, 0, padh))

        image_resize = torch.squeeze(F.interpolate(pad_image, size=(1024, 1024), mode='bilinear'),dim=0) / 255.0
        label_resize = torch.squeeze(F.interpolate(pad_label, size=(1024, 1024), mode='nearest'),dim=0)
        gt_temp = torch.squeeze(F.interpolate(gt_temp.float(), size=(1024, 1024), mode='nearest'),dim=0)

        return {'imidx':imidx,'image':image_resize, 'name':sample['name'], 'label':label_resize, 'gt_temp':gt_temp, 'initial_gt':initial_gt,
                'shape':torch.tensor(self.size), "ori_shape":shape, "ori_image":image.squeeze(0), "pad_image":pad_image, "padh":padh, "padw":padw}

class RandomCrop(object):
    def __init__(self,size=[288,288]):
        self.size = size
    def __call__(self,sample):
        imidx, image, label, shape =  sample['imidx'], sample['image'], sample['label'], sample['shape']

        h, w = image.shape[1:]
        new_h, new_w = self.size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[:,top:top+new_h,left:left+new_w]
        label = label[:,top:top+new_h,left:left+new_w]

        return {'imidx':imidx,'image':image, 'label':label, 'shape':torch.tensor(self.size)}


class Normalize(object):
    def __init__(self, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]):
        self.mean = mean
        self.std = std

    def __call__(self,sample):

        imidx, image, label, shape =  sample['imidx'], sample['image'], sample['label'], sample['shape']
        image = normalize(image,self.mean,self.std)

        return {'imidx':imidx,'image':image, 'label':label, 'shape':shape}



class LargeScaleJitter(object):
    """
        implementation of large scale jitter from copy_paste
        https://github.com/gaopengcuhk/Pretrained-Pix2Seq/blob/7d908d499212bfabd33aeaa838778a6bfb7b84cc/datasets/transforms.py
    """

    def __init__(self, output_size=1024, aug_scale_min=0.1, aug_scale_max=2.0):
        self.desired_size = torch.tensor(output_size)
        self.desired_size_surrogate = torch.tensor(512)
        self.aug_scale_min = aug_scale_min
        self.aug_scale_max = aug_scale_max

    def pad_target(self, padding, target):
        target = target.copy()
        if "masks" in target:
            target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[1], 0, padding[0]))
        return target

    def __call__(self, sample):
        imidx, image, label, image_size =  sample['imidx'], sample['image'], sample['label'], sample['shape']

        #resize keep ratio
        out_desired_size = (self.desired_size * image_size / max(image_size)).round().int()

        random_scale = torch.rand(1) * (self.aug_scale_max - self.aug_scale_min) + self.aug_scale_min
        scaled_size = (random_scale * self.desired_size).round()

        scale = torch.minimum(scaled_size / image_size[0], scaled_size / image_size[1])
        scaled_size = (image_size * scale).round().long()

        scaled_image = torch.squeeze(F.interpolate(torch.unsqueeze(image,0),scaled_size.tolist(),mode='bilinear'),dim=0)
        scaled_label = torch.squeeze(F.interpolate(torch.unsqueeze(label,0),scaled_size.tolist(),mode='bilinear'),dim=0)

        # random crop
        crop_size = (min(self.desired_size, scaled_size[0]), min(self.desired_size, scaled_size[1]))

        margin_h = max(scaled_size[0] - crop_size[0], 0).item()
        margin_w = max(scaled_size[1] - crop_size[1], 0).item()
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + crop_size[0].item()
        crop_x1, crop_x2 = offset_w, offset_w + crop_size[1].item()

        scaled_image = scaled_image[:,crop_y1:crop_y2, crop_x1:crop_x2]
        scaled_label = scaled_label[:,crop_y1:crop_y2, crop_x1:crop_x2]

        # pad
        padding_h = max(self.desired_size - scaled_image.size(1), 0).item()
        padding_w = max(self.desired_size - scaled_image.size(2), 0).item()
        image = F.pad(scaled_image, [0,padding_w, 0,padding_h],value=0) / 255.0
        label = F.pad(scaled_label, [0,padding_w, 0,padding_h],value=0)

        resize_image = torch.squeeze(F.interpolate(torch.unsqueeze(image,0), scale_factor=0.5,mode='bilinear'),dim=0)
        resize_label = torch.squeeze(F.interpolate(torch.unsqueeze(label,0), scale_factor=0.5,mode='bilinear'),dim=0)
        if "classes" in sample.keys():
            return {'imidx':imidx,'image':image, 'label':label, 'resize_image':resize_image, 'resize_label':resize_label,
                    'shape':torch.tensor(image.shape[-2:]), 'name':sample['name'], 'classes':sample['classes']}
        return {'imidx':imidx,'image':image, 'label':label, 'resize_image':resize_image, 'resize_label':resize_label,
                'shape':torch.tensor(image.shape[-2:]), 'name':sample['name']}


class OnlineDataset(Dataset):
    def __init__(self, name_im_gt_list, transform=None, eval_ori_resolution=False):

        self.transform = transform
        self.dataset = {}
        ## combine different datasets into one
        dataset_names = []
        dt_name_list = [] # dataset name per image
        im_name_list = [] # image name
        im_path_list = [] # im path
        gt_path_list = [] # gt path
        im_ext_list = [] # im ext
        gt_ext_list = [] # gt ext
        for i in range(0,len(name_im_gt_list)):
            dataset_names.append(name_im_gt_list[i]["dataset_name"])
            # dataset name repeated based on the number of images in this dataset
            dt_name_list.extend([name_im_gt_list[i]["dataset_name"] for x in name_im_gt_list[i]["im_path"]])
            im_name_list.extend([x.split(os.sep)[-1].split(name_im_gt_list[i]["im_ext"])[0] for x in name_im_gt_list[i]["im_path"]])
            im_path_list.extend(name_im_gt_list[i]["im_path"])
            gt_path_list.extend(name_im_gt_list[i]["gt_path"])
            im_ext_list.extend([name_im_gt_list[i]["im_ext"] for x in name_im_gt_list[i]["im_path"]])
            gt_ext_list.extend([name_im_gt_list[i]["gt_ext"] for x in name_im_gt_list[i]["gt_path"]])


        self.dataset["data_name"] = dt_name_list
        self.dataset["im_name"] = im_name_list
        self.dataset["im_path"] = im_path_list
        self.dataset["ori_im_path"] = deepcopy(im_path_list)
        self.dataset["gt_path"] = gt_path_list
        self.dataset["ori_gt_path"] = deepcopy(gt_path_list)
        self.dataset["im_ext"] = im_ext_list
        self.dataset["gt_ext"] = gt_ext_list

        self.eval_ori_resolution = eval_ori_resolution

    def __len__(self):
        return len(self.dataset["im_path"])
    def __getitem__(self, idx):
        im_path = self.dataset["im_path"][idx]
        gt_path = self.dataset["gt_path"][idx]
        im = io.imread(im_path)
        gt = io.imread(gt_path)
        if len(gt.shape) > 2:
            gt = gt[:, :, 0]
        if len(im.shape) < 3:
            im = im[:, :, np.newaxis]
        if im.shape[2] == 1:
            im = np.repeat(im, 3, axis=2)
        im = torch.tensor(im.copy(), dtype=torch.float32)
        im = torch.transpose(torch.transpose(im,1,2),0,1)
        gt = torch.unsqueeze(torch.tensor(gt, dtype=torch.float32),0)

        sample = {
        "imidx": torch.from_numpy(np.array(idx)),
        "image": im,
        "name": self.dataset["im_name"][idx],
        "gt_path": gt_path,
        "label": gt,
        "shape": torch.tensor(im.shape[-2:])
        }

        if self.transform:
            sample = self.transform(sample)

        if self.eval_ori_resolution:
            sample["ori_label"] = gt.type(torch.uint8)  # NOTE for evaluation only. And no flip here
            sample['ori_im_path'] = self.dataset["im_path"][idx]
            sample['ori_gt_path'] = self.dataset["gt_path"][idx]

        return sample

