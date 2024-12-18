export DETECTRON2_DATASETS=/remote-home/sunye/SAN_Test/datasets
export PYTHONPATH=/home/ubuntu/sunye/project/video_project/UnSeg_re:$PYTHONPATH
export model_name=unseg
export CUDA_VISIBLE_DEVICES="0,1"
export dataset=coco
python -m torch.distributed.launch --nproc_per_node=2 generate/generate_coco.py --gpu_ids 0,1,2,3,4,5,6,7 \
                --img_dir ./visualize/${dataset}/ues_${model_name} \
                --noise_dir ./visualize/${dataset}/noise \
                --infer_mode all \
                --target_class 255.0 \
                --checkpoint ./weight_dir/Generator_${model_name}_final.pth \
                --model-type vit_b \
                --noise_type min \
                --epsilon 0.0313725490196078 \
                --epsilon_bg 0.007843137254902 \
                --poison_datasets coco \
                --output work_dirs/hq_sam_b \
