#!/bin/bash
python train_consistency.py\
    "flowers"\
    "resnet50"\
    --noise_sd 0.5\
    --workers 4\
    --lr 0.01\
    --lr_step_size 50\
    --epochs 150\
    --batch 64\
    --num-noise-vec 2\
    --lbd 5\
    --eta 0.5\
    --pretrained-model "pretrained_resnet50.pth"\
    --exp-id "finetuned_model"