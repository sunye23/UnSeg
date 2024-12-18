python seg/train_seg.py --gpu_ids 0,1 \
                    --img_path image_aug \
                    --batch_size 16 \
                    --seg_pgt_path label_aug \
                    --list_path voc12/train_aug_id.txt \
                    --session_name model_clean_  > result_logs/model_clean_train.txt 2>&1

python seg/infer_seg.py --weights save_dir/model_clean_final.pth \
                        --save_path_c save_result/model_clean > result_logs/model_clean_infer.txt 2>&1

exit 0
