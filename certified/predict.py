# this file is based on code publicly available at
#   https://github.com/locuslab/smoothing
# written by Jeremy Cohen.

""" This script loads a base classifier and then runs PREDICT on many examples from a dataset."""
import argparse
import datetime
from time import time

import torch

from architectures import get_architecture
from transfer_datasets import get_dataset, DATASETS, get_num_classes
from train_utils import init_logger, requires_grad_

from third_party.smoothadv import Attacker, PGD_L2, DDN 
from third_party.core import Smooth

parser = argparse.ArgumentParser(description='Predict on many examples')
parser.add_argument("dataset", choices=DATASETS, help="which dataset")
parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("sigma", type=float, help="noise hyperparameter")
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")

#####################
# Attack params
parser.add_argument('--attack', default=None, type=str, choices=['DDN', 'PGD'])
parser.add_argument('--epsilon', default=64.0, type=float)
parser.add_argument('--num-steps', default=100, type=int)
parser.add_argument('--warmup', default=1, type=int)
parser.add_argument('--num-noise-vec', default=1, type=int,
                    help="number of noise vectors to use for finding adversarial examples")
parser.add_argument('--no-grad-attack', action='store_true',
                    help="Choice of whether to use gradients during attack or do the cheap trick")
parser.add_argument('--base-attack', action='store_true')
parser.add_argument("--visualize-examples", action='store_true', help="Whether to save the adversarial examples or not")

# PGD-specific
parser.add_argument('--random-start', default=True, type=bool)

# DDN-specific
parser.add_argument('--init-norm-DDN', default=256.0, type=float)
parser.add_argument('--gamma-DDN', default=0.05, type=float)


args = parser.parse_args()
assert 'checkpoint.pth.tar' in args.base_classifier
args.outdir = args.base_classifier.replace('checkpoint.pth.tar', '')

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


if __name__ == "__main__":

    if args.attack:
        if args.base_attack:
            outfile = '{}/{}_{}_base.log'.format(args.outdir, args.attack, args.epsilon, args.sigma)
        else:
            outfile = '{}/{}_{}_noise_{:.3}_mtest_{}.log'.format(args.outdir, args.attack, args.epsilon, args.sigma, args.num_noise_vec)
    
    else:
        outfile = '{}/noise_{:.3}.log'.format(args.outdir, args.sigma)

    args.epsilon /= 256.0
    args.init_norm_DDN /= 256.0
    if args.epsilon > 0:
        args.gamma_DDN = 1 - (3/510/args.epsilon)**(1/args.num_steps)

    # load the base classifier
    checkpoint = torch.load(args.base_classifier)
    base_classifier = get_architecture(checkpoint["arch"], args.dataset, False)
    base_classifier.load_state_dict(checkpoint['state_dict'])
    base_classifier.eval()
    requires_grad_(base_classifier, False)

    logger = init_logger(outfile)
    logger.info(args)
    logger.info(f"Loading from: {args.base_classifier} (epoch: {checkpoint['epoch']})")

    # create the smoothed classifier g
    smoothed_classifier = Smooth(base_classifier, get_num_classes(args.dataset), args.sigma)

    # prepare output file
    logger.info("idx\tlabel\tpredict\tbasePredict\tcorrect\ttime")

    if args.attack == 'PGD':
        logger.info('Attacker is PGD')
        attacker = PGD_L2(steps=args.num_steps, device='cuda', max_norm=args.epsilon)
    elif args.attack == 'DDN':
        logger.info('Attacker is DDN')
        attacker = DDN(steps=args.num_steps, device='cuda', max_norm=args.epsilon, 
                    init_norm=args.init_norm_DDN, gamma=args.gamma_DDN)

    # iterate through the dataset
    dataset = get_dataset(args.dataset, args.split)
    base_smoothed_agree = 0

    start_img = 0
    num_img = int(dataset.__len__()/args.skip)
    count = 0
    total = 0
    for i in range(num_img):
        before_time = time()
        x, label = dataset[start_img + i * args.skip]
        x = x.unsqueeze(0).cuda()
        label = torch.Tensor([label]).cuda().long()

        if args.attack in ['PGD', 'DDN']:
            x = x.repeat((args.num_noise_vec, 1, 1, 1))

            noise = (1 - int(args.base_attack))* torch.randn_like(x, device='cuda') * args.sigma
            x = attacker.attack(base_classifier, x, label, 
                                noise=noise, num_noise_vectors=args.num_noise_vec,
                                no_grad=args.no_grad_attack,
                                )
            x = x[:1]

        base_output = base_classifier(x)
        base_prediction = base_output.argmax(1).item()
        
        x = x.squeeze()
        label = label.item()
        # make the prediction
        prediction = smoothed_classifier.predict(x, args.N, args.alpha, args.batch)
        if base_prediction == prediction:
            base_smoothed_agree += 1 
        after_time = time()
        correct = int(prediction == label)
        count += correct
        total += 1

        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))

        # log the prediction and whether it was correct
        logger.info("{}\t{}\t{}\t{}\t{}\t{}".format(i, label, prediction, base_prediction, correct, time_elapsed))

    logger.info(f'PGD accuracy: {100*count/total:.5f}%')
