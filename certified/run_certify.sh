#!/bin/bash
DS='flowers'
SD=0.5
EXP='finetuned_model'

python certify.py $DS\
    "checkpoints/consistency/${DS}/cohen/num_2/lbd_5.0/eta_0.5/noise_${SD}/${EXP}/resnet50/checkpoint.pth.tar"\
    $SD\
    --alpha 0.001\
    --N0 100\
    --N 100000\
    --compute_acr\
    --skip 12\
    --verbose
