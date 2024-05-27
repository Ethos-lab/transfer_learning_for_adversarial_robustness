# this file is based on code publicly available at
#   https://github.com/locuslab/smoothing
# written by Jeremy Cohen.

""" This script loads a base classifier and then runs PREDICT on many examples from a dataset."""
import os
import argparse

import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

from architectures import get_architecture
from datasets import get_dataset, DATASETS
from train_utils import test, init_logger


seeds = [1111,2222,3333]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser(description='Predict on many examples')
parser.add_argument("--dataset", choices=DATASETS, help="which dataset")
parser.add_argument("--base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("--noise", type=str, default='gaussian', choices=['gaussian', 'uniform'])
parser.add_argument("--sigma", type=float, help="noise hyperparameter")
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--workers", type=int, default=4, help="for dataloader")
parser.add_argument("--trials", type=int, default=1, help="num eval trials")

args = parser.parse_args()


def main():
    if 'checkpoint.pth.tar' in args.base_classifier:
        args.outfile = args.base_classifier.replace('checkpoint.pth.tar', f'noise_eval_s{args.sigma}.log')
    else:
        term = args.base_classifier.split('/')[-1]
        assert '.pth' in term
        save_dir = args.base_classifier.replace(term, term.split('.')[0])
        os.makedirs(save_dir, exist_ok=True)
        args.outfile = f'{save_dir}/noise_eval_s{args.sigma}.log'

    # prepare output file
    logger = init_logger(args.outfile)
    logger.info(args)

    # load the base classifier
    checkpoint = torch.load(args.base_classifier)
    model = get_architecture(checkpoint["arch"], args.dataset)
    model.load_state_dict(checkpoint['state_dict'])
    try:
        epoch = checkpoint["epoch"]
    except KeyError:
        epoch = -1
    logger.info(f'Loaded from: {args.base_classifier} (epoch: {epoch})')

    dataset = get_dataset(args.dataset, args.split)
    pin_memory = (args.dataset == "imagenet")
    loader = DataLoader(dataset, shuffle=False, batch_size=args.batch,
                        num_workers=args.workers, pin_memory=pin_memory)

    criterion = CrossEntropyLoss().to(device)

    for t_idx in range(args.trials):
        torch.manual_seed(seeds[t_idx])
        _, acc = test(loader, model, criterion, -1, args.sigma, device, writer=None, noise_type=args.noise)
        logger.info(f"Accuracy (trial {t_idx}): {acc}%")


if __name__ == "__main__":
    main()

