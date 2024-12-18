export PYTHONPATH=/home/ubuntu/sunye/project/video_project/UnSeg_re:$PYTHONPATH
export DETECTRON2_DATASETS=UnSeg/datasets     # Path to the datasets, such as coco, ade20k, cityscapes, refer to: https://github.com/facebookresearch/Mask2Former/blob/main/datasets/README.md
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

python -m torch.distributed.launch --nproc_per_node=8 \
	train.py \
	--checkpoint ./pretrained_checkpoint/sam_vit_b_01ec64.pth \
	--model-type vit_b \
	--model_name BaseModel \
	--generator_mode label_modify \
	--epsilon 0.003921568627451 \
	--epsilon_bg 9.803921568627451e-4 \
	--output work_dirs/hq_sam_b