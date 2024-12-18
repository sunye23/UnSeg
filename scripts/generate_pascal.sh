export DETECTRON2_DATASETS=/remote-home/sunye/SAN_Test/datasets
export PYTHONPATH=/home/ubuntu/sunye/project/video_project/UnSeg_re:$PYTHONPATH
export model_name=unseg
export dataset=pascal
python -m torch.distributed.launch --nproc_per_node=1 generate.py --gpu_ids 0 \
                --img_dir ./result/visualize/${dataset}/ues_${model_name} \
                --noise_dir ./result/visualize/${dataset}/noise \
                --infer_mode all \
                --target_class 6.0 \
                --checkpoint ./weight_dir/Generator_${model_name}_final.pth \
                --model-type vit_b \
                --noise_type min \
                --epsilon 0.0313725490196078 \
                --epsilon_bg 0.007843137254902 \
                --poison_datasets pascal \
                --output work_dirs/hq_sam_b \
