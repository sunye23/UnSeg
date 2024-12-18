export DETECTRON2_DATASETS=/remote-home/sunye/SAN_Test/datasets
export PYTHONPATH=/home/ubuntu/sunye/project/video_project/UnSeg_re:$PYTHONPATH
# for lung segmentation datasets
export model_name=unseg
export dataset=Lung_seg
python -m torch.distributed.launch --nproc_per_node=8 generate/generate_medical.py --gpu_ids 0,1,2,3,4,5,6,7 \
                --img_dir ./visualize/${dataset}/ues_${model_name} \
                --noise_dir ./visualize/${dataset}/noise \
                --source_img /remote-home/sunye/test/medical_seg/LungSegmentation/train \
                --source_gt /remote-home/sunye/test/medical_seg/LungSegmentation/trainannot \
                --checkpoint ./weight_dir/Generator_${model_name}_final.pth \
                --model-type vit_b \
                --image_type .png \
                --epsilon 0.0156862745098039 \
                --epsilon_bg 0.003921568627451 \
                --poison_datasets medical \
                --output work_dirs/hq_sam_b \

exit 0

export model_name=unseg
python -m torch.distributed.launch --nproc_per_node=8 generate/generate_medical.py --gpu_ids 0,1,2,3 \
                --img_dir /remote-home/sunye/test/medical_seg/LungSegmentation/Lung_train_poi4 \
                --noise_dir /remote-home/sunye/test/medical_seg/LungSegmentation/Lung_noise_poi4 \
                --source_img /remote-home/sunye/test/medical_seg/LungSegmentation/train \
                --source_gt /remote-home/sunye/test/medical_seg/LungSegmentation/trainannot \
                --checkpoint ./weight_dir/Generator_${model_name}_final.pth \
                --model-type vit_b \
                --image_type .png \
                --epsilon 0.0156862745098039 \
                --epsilon_bg 0.003921568627451 \
                --poison_datasets medical \
                --output work_dirs/hq_sam_b \


# for Kvasir_seg datasets
#export model_name=unseg
#export dataset=Kvasir_seg
#python -m torch.distributed.launch --nproc_per_node=8 generate/generate_medical.py --gpu_ids 0,1,2,3,4,5,6,7 \
#                --img_dir ./visualize/${dataset}/ues_${model_name} \
#                --noise_dir ./visualize/${dataset}/noise \
#                --source_img /remote-home/sunye/test/medical_seg/Kvasir-SEG/train \
#                --source_gt /remote-home/sunye/test/medical_seg/Kvasir-SEG/trainannot \
#                --checkpoint ./weight_dir/Generator_${model_name}_final.pth \
#                --model-type vit_b \
#                --image_type .jpg \
#                --epsilon 0.0156862745098039 \
#                --epsilon_bg 0.003921568627451 \
#                --poison_datasets medical \
#                --output work_dirs/hq_sam_b \

exit 0

export model_name=unseg
python -m torch.distributed.launch --nproc_per_node=6 generate/generate_medical.py --gpu_ids 1,2,3,4,5,6 \
                --img_dir ./visualize/${dataset}/ues_${model_name} \
                --noise_dir ./visualize/${dataset}/noise \
                --source_img /remote-home/sunye/test/medical_seg/Kvasir-SEG/train \
                --source_gt /remote-home/sunye/test/medical_seg/Kvasir-SEG/trainannot \
                --checkpoint ./weight_dir/Generator_${model_name}_final.pth \
                --model-type vit_b \
                --image_type .jpg \
                --epsilon 0.0313725490196078 \
                --epsilon_bg 0.007843137254902 \
                --poison_datasets medical \
                --output work_dirs/hq_sam_b \

