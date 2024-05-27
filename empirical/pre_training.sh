#!/bin/bash
DS="flowers"
ARCH="resnet50"

python -m robustness.transfer_main --arch $ARCH \
    --dataset $DS \
    --data "data/flowers" \
    --out-dir "checkpoints" \
    --exp-name ${DS}_${ARCH}_finetuned \
    --adv-train 1 \
    --adv-eval 1 \
    --constraint 2 \
    --attack-steps 3 \
    --eps 0.5 \
    --attack-lr 0.33334 \
    --random-start 1 \
    --batch-size 64 \
    --weight-decay 0.0001 \
    --lr 0.01 \
    --log-iters 1 \
    --workers 4 \
    --model-path "pretrained_resnet50.pth" \
    --freeze-level -1