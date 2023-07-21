DATA_DIR="/home/daoduyhung/hicehehe/iccv2023/data"
EXPERIMENT_NAME="experiment7"

python3 rewrite_pretrain.py \
    --output_dir ./output_dir/$EXPERIMENT_NAME \
    --log_dir ./output_dir/$EXPERIMENT_NAME \
    --batch_size 128 \
    --model mae_vit_base_patch16 \
    --mask_ratio 0.75 \
    --epochs 1600 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path $DATA_DIR \
    --num_workers 2
