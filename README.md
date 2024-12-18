
## 1. Installation
Clone this repository and navigate to project folder

```bash
git clone https://github.com/sunye23/UnSeg.git
cd UnSeg
```

Quick Installation **(Ensure your CUDA version is correct before installation; we use torch==1.13.1+cu117.)**

```Shell
conda create --name unseg python=3.8 -y
conda activate unseg
bash install.sh
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2
pip install -e .
pip install git+https://github.com/cocodataset/panopticapi.git
pip install git+https://github.com/mcordts/cityscapesScripts.git
```

We organize the training folder as follows.
```
Project
|____data
| |____cascade_psp
| |____DIS5K
| |____thin_object_detection
|____eval
|____scripts
|____generate
|____weight_dir
| |____Generator_unseg_final.pth
|____pretrained_checkpoint
| |____modified_mae_pretrain_vit_base.pth
| |____sam_vit_b_backbone.pth
| |____sam_vit_b_maskdecoder.pth
| |____sam_vit_b_prompt_decoder.pth
|____train.py
|____utils
|____segment_anything_training
|____work_dirs
```
## 2. Training Data Preparation

Our training data is based on HQSeg-44K and can be downloaded from [hugging face link](https://huggingface.co/sam-hq-team/sam-hq-training/tree/main/data)

## Expected dataset structure for HQSeg-44K

```
data
|____DIS5K
|____cascade_psp
| |____DUTS-TE
| |____DUTS-TR
| |____ecssd
| |____fss_all
| |____MSRA_10K
|____thin_object_detection
| |____COIFT
| |____HRSOD
| |____ThinObject5K
```

## 3. Pretrained checkpoint Preparation

Download the [model weight](https://pan.baidu.com/s/1cGmfyNAlbE3Z0NFjsCr92w?pwd=7dur ) here and put them into right dir.

## 4. Training
Model training is based on RTX3090 x 8.

```Shell
bash scripts/run_train.sh
```

## 5. UEs Generation
UEs generation is based on RTX3090 x 8.

Before generating ues, place the clean images and the corresponding groundtruths in the correct directory. For the datasets such as coco, cityscapes, and ade20k, please refer to [Mask2former](https://github.com/facebookresearch/Mask2Former) for data preparation. 

Here we only provide the download guidance for the pascal voc dataset.

<details>
<summary>
PASCAL VOC 2012
</summary>

- Download [the PASCAL VOC 2012 development kit](http://host.robots.ox.ac.uk/pascal/VOC/voc2012).
  ``` bash
  wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
  tar –xvf VOCtrainval_11-May-2012.tar
  ```
- Download augmented annoations `SegmentationClassAug.zip` from [SBD dataset](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6126343&casa_token=cOQGLW2KWqUAAAAA:Z-QHpQPf8Pnb07A75yBm2muYjqJwYUYPFbwwxMFHRcjRX0zl45kEGNqyTEPH7irB2QbabZbn&tag=1) via this [link](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0).
- Make your data directory like this below
  ``` bash
  VOCdevkit/
  └── VOC2012
      ├── Annotations
      ├── ImageSets
      ├── JPEGImages
      ├── SegmentationClass
      ├── SegmentationClassAug
      └── SegmentationObject
    ```

  </details>
  
For Pascal VOC dataset. 
```Shell
bash scripts/generate_pascal.sh
```

For Ade20K dataset. 
```Shell
bash scripts/generate_ade20k.sh
```

For COCO dataset. 
```Shell
bash scripts/generate_coco.sh
```

For Cityscapes dataset. 
```Shell
bash scripts/generate_cityscapes.sh
```

For interactive dataset. 
```Shell
bash scripts/generate_interactive.sh
```

For remote sensing dataset, please refer to [Rsprompter](https://github.com/KyanChen/RSPrompter) for data preparation. 
```Shell
bash scripts/generate_remote.sh
```

For medical image dataset.
```Shell
bash scripts/generate_medical.sh
```
## 6. Evaluation

For the DeepLabv1 evaluation, we mainly employ the code provided by [MCTFormer](https://github.com/xulianuwa/MCTformer). You should follow the work to prepare the weights and environments. We also provide the code and the script under the eval dir. You could the command following:
```Shell
bash eval/run_seg.sh
```

For mainstream image segmentation evaluation, please refer to [Mask2former](https://github.com/facebookresearch/Mask2Former).

For interactive image segmentation evaluation, please refer to [SAM-HQ](https://github.com/SysCV/sam-hq).

For remote sensing evaluation, please refer to [Rsprompter](https://github.com/KyanChen/RSPrompter).

For medical image segmentation evaluation, our code is based on the **segmentation_models_pytorch** codebase.

For object detection evaluation, please refer to [DINO](https://github.com/IDEA-Research/DINO).
## Citation
If you find UnSeg useful for your research and applications, please cite using this BibTeX:
```bibtex
@inproceedings{sun2024unseg,
  title={UnSeg: One Universal Unlearnable Example Generator is Enough against All Image Segmentation},
  author={Sun, Ye and Zhang, Hao and Zhang, Tiehua and Ma, Xingjun and Jiang, Yu-Gang},
  booktitle={NeurIPS},
  year={2024}
}
```
