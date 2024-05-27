GPUS=5
CUDA_VISIBLE_DEVICES=$GPUS python eval.py --data_dir ${HOME}/data --norm L2 --epsilon 0.5 --save_dir ${HOME}/checkpoints/adv_training/cifar10_224_resnet50_l2_0_5_3steps_bs128_while --batch_size 500 --version custom --individual
