export PYTHONPATH=/home/ubuntu/sunye/project/video_project/UnSeg_re:$PYTHONPATH
export DETECTRON2_DATASETS=/remote-home/sunye/SAN_Test/datasets # Path to the datasets, such as coco, ade20k, cityscapes, refer to: https://github.com/facebookresearch/Mask2Former/blob/main/datasets/README.md
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export model_name=unseg
export dataset=Ade20K
python -m torch.distributed.launch --nproc_per_node=8 generate/generate_ade20k.py --gpu_ids 0,1,2,3,4,5,6,7 \
                --img_dir ./visualize/${dataset}/ues_${model_name} \
                --noise_dir ./visualize/${dataset}/noise \
                --infer_mode all \
                --target_class 255.0 \
                --target_list 12 15 20 25 36 66 80 90 98 115 119 127 142 143 148 \
                --checkpoint ./weight_dir/Generator_${model_name}_final.pth \
                --model-type vit_b \
                --noise_type min \
                --epsilon 0.0313725490196078 \
                --epsilon_bg 0.007843137254902 \
                --poison_datasets ade20k \
                --output work_dirs/hq_sam_b \
