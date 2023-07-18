DATA_DIR="/home/daoduyhung/hicehehe/iccv2023/ICCV_classification_data"
EXPERIMENT_NAME="linprobe_experiment1"
PRETRAIN_CHKPT="/home/daoduyhung/hicehehe/iccv2023/mae/output_dir/experiment7/checkpoint-1599.pth"

python3 rewrite_linprobe.py \
    --output_dir ./output_dir/$EXPERIMENT_NAME \
    --log_dir ./output_dir/$EXPERIMENT_NAME \
    --batch_size 64 \
    --model vit_base_patch16 --cls_token \
    --finetune $PRETRAIN_CHKPT \
    --epochs 50 \
    --blr 2.2e-2 --weight_decay 0.0 \
    --data_path $DATA_DIR \
    --num_workers 4 \
    --nb_classes 3 \
    --seed 23
