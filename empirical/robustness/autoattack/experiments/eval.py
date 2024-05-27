import os
import argparse
from pathlib import Path
import warnings

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms

import sys
sys.path.insert(0,'..')
sys.path.insert(0, '/home/pvaishnavi/projects/robustness/robustness')

from model_utils import make_and_restore_model
from datasets import DATASETS


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='imagenet')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--norm', type=str, default='Linf')
    parser.add_argument('--epsilon', type=float, default=8./255.)
    parser.add_argument('--individual', action='store_true')
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--version', type=str, default='standard')

    args.log_path = f'{args.save_dir}/adv_eval.txt'
    args.model = f'{args.save_dir}/checkpoint.pt.best'
    assert os.path.exists(args.model), 'checkpoint file does not exist !!!'

    args = parser.parse_args()

    return args


def main():
    args = get_args()
    print(args)

    # load model
    classifier_model = dataset.get_model(arch, pytorch_pretrained) if \
                            isinstance(arch, str) else arch


    ckpt = torch.load(args.model)['model']
    filtered_ckpt = {k.replace('module.model.', ''): v for k,v in ckpt.items() if k.startswith('module.model.')}
    model.load_state_dict(filtered_ckpt)
    del ckpt
    model.to('cuda:0')
    model.eval()

    # load data
    test_dataset = get_dataset(args.dataset, 'test')
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch,
                             num_workers=args.workers, pin_memory=pin_memory)


    # create save dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load attack
    from autoattack import AutoAttack
    adversary = AutoAttack(model, norm=args.norm, eps=args.epsilon, log_path=args.log_path,
        version=args.version)

    # example of custom version
    if args.version == 'custom':
        adversary.attacks_to_run = ['apgd-ce']
        adversary.apgd.n_restarts = 1

    # run attack and save images
    with torch.no_grad():
        for inputs, targets in test_loader:
            if not args.individual:
                adv_inputs = adversary.run_standard_evaluation(inputs, targets, bs=args.batch_size)

            else:
                # individual version, each attack is run on all test points
                adv_inputs = adversary.run_standard_evaluation_individual(inputs, targets, bs=args.batch_size)



if __name__ == '__main__':
    main()
