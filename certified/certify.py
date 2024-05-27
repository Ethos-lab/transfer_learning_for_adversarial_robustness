# this file is based on code publicly available at
#   https://github.com/locuslab/smoothing
# written by Jeremy Cohen.

""" Evaluate a smoothed classifier on a dataset. """
import argparse
import os
import datetime
from time import time
import scipy.io as sio
import numpy as np
import tqdm
import torch
import copy

from third_party.core import Smooth
from architectures import get_architecture
from transfer_datasets import get_dataset, DATASETS, get_num_classes
from train_utils import init_logger


parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("dataset", choices=DATASETS, help="which dataset")
parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("sigma", type=float, help="noise hyperparameter")
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--compute_acr", action="store_true")
parser.add_argument("--trials", type=int, default=1)

args = parser.parse_args()


IMAGENET_GRID = (0.50, 1.00, 1.50, 2.00, 2.50, 3.00)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def acr_certify(smooth_net, dataset, matfile, logger):
    grid = IMAGENET_GRID
    start_img = 0
    num_img = int(dataset.__len__()/args.skip)

    logger.info('===certify(N={}, sigma={})==='.format(args.N, args.sigma))
    radius_hard = np.zeros((num_img,), dtype=np.float)
    num_grid = len(grid)
    cnt_grid_hard = np.zeros((num_grid + 1,), dtype=np.int)
    s_hard = 0.0

    pbar = tqdm.tqdm(total=num_img, leave=False)
    for i in range(num_img):
        img, target = dataset[start_img + i * args.skip]
        img = img.to(device)

        prediction, radius = smooth_net.certify(img, args.N0, args.N, args.alpha, args.batch)
        correct = int(prediction == target)
        if args.verbose:
            if correct == 1:
                print('Correct: 1. Radius: {}.'.format(radius))
            else:
                print('Correct: 0.')
        radius_hard[i] = radius if correct == 1 else -1
        if correct == 1:
            cnt_grid_hard[0] += 1
            s_hard += radius
            for j in range(num_grid):
                if radius >= grid[j]:
                    cnt_grid_hard[j + 1] += 1

        pbar.update(1)

    logger.info('===Certify Summary===')
    logger.info('Total Image Number: {}'.format(num_img))
    logger.info('Radius: 0.0  Number: {}  Acc: {}'.format(
        cnt_grid_hard[0], cnt_grid_hard[0] / num_img * 100))
    for j in range(num_grid):
        logger.info('Radius: {}  Number: {}  Acc: {}'.format(
          grid[j], cnt_grid_hard[j + 1], cnt_grid_hard[j + 1] / num_img * 100))
    logger.info('ACR: {}'.format(s_hard / num_img))
    if matfile is not None:
        sio.savemat(matfile, {'hard': radius_hard})


def basic_certify(smooth_net, dataset, logger):
    logger.info("idx\tlabel\tpredict\tradius\tcorrect\ttime")
    # iterate through the dataset
    for i in range(len(dataset)):
        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i == args.max:
            break

        (x, label) = dataset[i]

        before_time = time()
        # certify the prediction of g around x
        x = x.to(device)
        prediction, radius = smooth_net.certify(x, args.N0, args.N, args.alpha, args.batch)
        after_time = time()
        correct = int(prediction == label)

        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        logger.info("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
            i, label, prediction, radius, correct, time_elapsed))


def main():
    # load the base classifier
    checkpoint = copy.deepcopy(torch.load(args.base_classifier, map_location=device))
    if 'state_dict' in checkpoint.keys():
        sd = checkpoint['state_dict']
    else: #robustness checkpoint
        sd = checkpoint['model']
        sd = {k.replace('module.model.', '1.module.'):v for k,v in sd.items() if ('attacker' not in k) and ('normalizer' not in k)}
        checkpoint['arch'] = 'resnet50'
        checkpoint['epoch'] = -1
    try:
        base_classifier = get_architecture(checkpoint["arch"], args.dataset, False)
        base_classifier.load_state_dict(sd)
        epoch = checkpoint['epoch']
    except KeyError:
        arch = 'resnet50' #input('enter n/w arch: ')
        base_classifier = get_architecture(arch, args.dataset, False)
        base_classifier.load_state_dict({f'1.module.{k}':v for k,v in sd.items()})
        epoch = -1

    # create the smooothed classifier g
    smooth_classifier = Smooth(base_classifier, get_num_classes(args.dataset), args.sigma)

    dataset = get_dataset(args.dataset, args.split)

    for tnum in range(args.trials):
        matfile = args.base_classifier.replace('checkpoint.pth.tar', f'certificate-{tnum}.mat')
        outfile = args.base_classifier.replace('checkpoint.pth.tar', f'certificate-{tnum}.log')
        logger = init_logger(outfile)
        logger.info(args)
        logger.info(f"Loading from: {args.base_classifier} (epoch: {epoch})")

        if args.compute_acr:
            acr_certify(smooth_classifier, dataset, matfile, logger)
        else:
            basic_certify(smooth_classifier, dataset, logger)


if __name__ == '__main__':
    main()
