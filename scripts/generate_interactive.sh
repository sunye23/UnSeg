export DETECTRON2_DATASETS=/remote-home/sunye/SAN_Test/datasets
export PYTHONPATH=/home/ubuntu/sunye/project/video_project/UnSeg_re:$PYTHONPATH
export model_name=unseg
export dataset=hqseg_44k
python -m torch.distributed.launch --nproc_per_node=8 generate/generate_interactive.py --gpu_ids 0,1,2,3,4,5,6,7  \
                --img_dir ./visualize/${dataset}/ues_${model_name} \
                --noise_dir ./visualize/${dataset}/noise \
                --source_img ./data/DIS5K/DIS-TR/im \
                --source_gt ./data/DIS5K/DIS-TR/gt \
                --checkpoint ./weight_dir/Generator_${model_name}_final.pth \
                --model-type vit_b \
                --epsilon 0.0313725490196078 \
                --epsilon_bg 0.007843137254902 \
                --poison_datasets interactive \
                --output work_dirs/hq_sam_b \

export model_name=unseg
python -m torch.distributed.launch --nproc_per_node=7 generate/generate_interactive.py --gpu_ids 0,1,2,3,4,5,6 \
                --img_dir ./HQSegV2/poison_ThinObject5kTR_eps8 \
                --noise_dir ./HQSegV2/noise_ThinObject5kTR_eps8 \
                --source_img ./data/thin_object_detection/ThinObject5K/images_train \
                --source_gt ./data/thin_object_detection/ThinObject5K/masks_train \
                --checkpoint ./weight_dir/Generator_${model_name}_final.pth \
                --model-type vit_b \
                --epsilon 0.0313725490196078 \
                --epsilon_bg 0.007843137254902 \
                --poison_datasets interactive \
                --output work_dirs/hq_sam_b \

export model_name=unseg
python -m torch.distributed.launch --nproc_per_node=7 generate/generate_interactive.py --gpu_ids 0,1,2,3,4,5,6 \
                --img_dir ./HQSegV2/poison_FSS_eps8 \
                --noise_dir ./HQSegV2/noise_FSS_eps8 \
                --source_img ./data/cascade_psp/fss_all \
                --source_gt ./data/cascade_psp/fss_all \
                --checkpoint ./weight_dir/Generator_${model_name}_final.pth \
                --model-type vit_b \
                --epsilon 0.0313725490196078 \
                --epsilon_bg 0.007843137254902 \
                --poison_datasets interactive \
                --output work_dirs/hq_sam_b \


export model_name=unseg
python -m torch.distributed.launch --nproc_per_node=7 generate/generate_interactive.py --gpu_ids 0,1,2,3,4,5,6 \
                --img_dir ./HQSegV2/poison_DUTSTR_eps8 \
                --noise_dir ./HQSegV2/noise_DUTSTR_eps8 \
                --source_img ./data/cascade_psp/DUTS-TR \
                --source_gt ./data/cascade_psp/DUTS-TR \
                --checkpoint ./weight_dir/Generator_${model_name}_final.pth \
                --model-type vit_b \
                --epsilon 0.0313725490196078 \
                --epsilon_bg 0.007843137254902 \
                --poison_datasets interactive \
                --output work_dirs/hq_sam_b \


export model_name=unseg
python -m torch.distributed.launch --nproc_per_node=7 generate/generate_interactive.py --gpu_ids 0,1,2,3,4,5,6 \
                --img_dir ./HQSegV2/poison_DUTSTE_eps8 \
                --noise_dir ./HQSegV2/noise_DUTSTE_eps8 \
                --source_img ./data/cascade_psp/DUTS-TE \
                --source_gt ./data/cascade_psp/DUTS-TE \
                --checkpoint ./weight_dir/Generator_${model_name}_final.pth \
                --model-type vit_b \
                --epsilon 0.0313725490196078 \
                --epsilon_bg 0.007843137254902 \
                --poison_datasets interactive \
                --output work_dirs/hq_sam_b \

export model_name=unseg
python -m torch.distributed.launch --nproc_per_node=7 generate/generate_interactive.py --gpu_ids 0,1,2,3,4,5,6 \
                --img_dir ./HQSegV2/poison_ECSSD_eps8 \
                --noise_dir ./HQSegV2/noise_ECSSD_eps8 \
                --source_img ./data/cascade_psp/ecssd \
                --source_gt ./data/cascade_psp/ecssd \
                --checkpoint ./weight_dir/Generator_${model_name}_final.pth \
                --model-type vit_b \
                --epsilon 0.0313725490196078 \
                --epsilon_bg 0.007843137254902 \
                --poison_datasets interactive \
                --output work_dirs/hq_sam_b \

export model_name=unseg
python -m torch.distributed.launch --nproc_per_node=7 generate/generate_interactive.py --gpu_ids 0,1,2,3,4,5,6 \
                --img_dir ./HQSegV2/poison_MSRA10K_eps8 \
                --noise_dir ./HQSegV2/noise_MSRA10K_eps8 \
                --source_img ./data/cascade_psp/MSRA_10K \
                --source_gt ./data/cascade_psp/MSRA_10K \
                --checkpoint ./weight_dir/Generator_${model_name}_final.pth \
                --model-type vit_b \
                --epsilon 0.0313725490196078 \
                --epsilon_bg 0.007843137254902 \
                --poison_datasets interactive \
                --output work_dirs/hq_sam_b \
