export DETECTRON2_DATASETS=/remote-home/sunye/SAN_Test/datasets
export PYTHONPATH=/home/ubuntu/sunye/project/video_project/UnSeg_re:$PYTHONPATH
export model_name=unseg
export dataset=NWPU
python -m torch.distributed.launch --nproc_per_node=8 generate/generate_remote.py --gpu_ids 0,1,2,3,4,5,6,7 \
                --img_dir ./visualize/${dataset}/ues_${model_name} \
                --noise_dir ./visualize/${dataset}/noise \
                --checkpoint ./weight_dir/Generator_${model_name}_final.pth \
                --model-type vit_b \
                --epsilon 0.0313725490196078 \
                --epsilon_bg 0.007843137254902 \
                --poison_datasets remote_sensing \
                --output work_dirs/hq_sam_b \
