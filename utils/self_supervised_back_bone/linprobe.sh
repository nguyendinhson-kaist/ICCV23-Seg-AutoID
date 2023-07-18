DATA_DIR="/home/daoduyhung/hicehehe/iccv2023/linprobe_data"
EXPERIMENT_NAME="linprobe_experiment1"
PRETRAIN_CHKPT="/home/daoduyhung/hicehehe/iccv2023/mae/output_dir/experiment7/checkpoint-1599.pth"

python3 rewrite_linprobe.py \
    --output_dir ./output_dir/$EXPERIMENT_NAME \
    --log_dir ./output_dir/$EXPERIMENT_NAME \
    --batch_size 32 \
    --model vit_base_patch16 --cls_token \
    --finetune $PRETRAIN_CHKPT \
    --epochs 50 \
    --warmup_epochs 5 \
    --blr 1e-3 --weight_decay 0.0 \
    --data_path $DATA_DIR \
    --num_workers 2 \
    --nb_classes 3 \
    --seed 23
