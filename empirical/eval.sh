#!/bin/bash
DS="flowers"
ARCH="resnet50"

python -m robustness.third_party_eval\
    --arch resnet50 \
    --dataset $DS \
    --data "data/flowers" \
    --workers 4 \
    --out-dir checkpoints \
    --exp-name ${DS}_${ARCH}_finetuned \
    --eval-only 1 \
    --adv-eval 0 \
    --eps 0.5\
    --batch-size 256\
    --constraint 2 \
    --attacks apgd-ce square \
    --attack_model