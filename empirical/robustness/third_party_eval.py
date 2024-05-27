import os.path
import sys

import tqdm
import argparse
import numpy as np

import torch
import torch as ch
import torch.nn as nn
from torchvision import models

from autoattack import AutoAttack

try:
    from .datasets import DATASETS
    from .transfer_datasets import TRANSFER_DATASETS
    from .model_utils import make_and_restore_model
    from .tools import helpers
    from .transfer_utils import constants as cs
    from .transfer_utils import fine_tunify
    from . import defaults
    from . import datasets
    from . import transfer_datasets
except:
    raise ValueError("Make sure to run with python -m from root project directory")


parser = argparse.ArgumentParser(description='Transfer learning via pretrained Imagenet models',
                                 conflict_handler='resolve')
parser = defaults.add_args_to_parser(defaults.CONFIG_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.MODEL_LOADER_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.TRAINING_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.PGD_ARGS, parser)

# Custom arguments
parser.add_argument('--dataset', type=str, default='cifar',
                    help='Dataset (Overrides the one in robustness.defaults)')
parser.add_argument('--model-path', type=str, default='')
parser.add_argument('--resume', action='store_true',
                    help='Whether to resume or not (Overrides the one in robustness.defaults)')
parser.add_argument('--pytorch-pretrained', action='store_true',
                    help='If True, loads a Pytorch pretrained model.')
parser.add_argument('--cifar10-cifar10', action='store_true',
                    help='cifar10 to cifar10 transfer')
parser.add_argument('--subset', type=int, default=None,
                    help='number of training data to use from the dataset')
parser.add_argument('--no-tqdm', type=int, default=1,
                    choices=[0, 1], help='Do not use tqdm.')
parser.add_argument('--no-replace-last-layer', action='store_true',
                    help='Whether to avoid replacing the last layer')
parser.add_argument('--freeze-level', type=int, default=-1,
                    help='Up to what layer to freeze in the pretrained model (assumes a resnet architectures)')
parser.add_argument('--additional-hidden', type=int, default=0,
                    help='How many hidden layers to add on top of pretrained network + classification layer')
parser.add_argument('--per-class-accuracy', action='store_true', help='Report the per-class accuracy. '
                    'Can be used only with pets, caltech101, caltech256, aircraft, and flowers.')

parser.add_argument("--attacks", nargs='+', default="pgd")
parser.add_argument("--attack_model", action='store_false')

device = torch.device('cuda:0') if torch.cuda.device_count() > 0 else torch.device('cpu')


def test(model, test_loader, attack=None):
    correct = 0
    total = 0
    pbar = tqdm.tqdm(total=len(test_loader), leave=False)
    for im, lbl in test_loader:
        im, lbl = im.to('cpu').numpy(), lbl.to('cpu').numpy()
        if attack:
            im_adv = attack.generate(im, lbl)
            pred = model.predict(im_adv)
        else:
            pred = model.predict(im)
        label_pred = np.argmax(pred, axis=1)
        correct += (label_pred == lbl).sum()
        total += im.shape[0]
        pbar.update(1)
    pbar.close()

    return 100*correct/total


def main(args):

    for arg, value in sorted(vars(args).items()):
        print(f"Argument {arg}: {value}")

    # Load dataset
    ds, _, test_loader = get_dataset_and_loaders(args)
    testset = test_loader.dataset
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=testset.__len__(), shuffle=True, num_workers=args.workers, drop_last=True)

    if torch.cuda.device_count() > 0:
        test_loader = helpers.DataPrefetcher(test_loader)

    model, checkpoint = get_model(args, ds)

    model.eval()
    log_path = f"{args.out_dir}/{args.exp_name}/{','.join(args.attacks)}_result.txt"
    runAA(model, test_loader, log_path, args.eps, args.batch_size)


def runAA(model, loader, log_path, epsilon, bs=100):
    model.eval()
    adversary = AutoAttack(model, norm=f'L{args.constraint}', eps=float(epsilon),
                           version='custom', log_path=log_path, attacks_to_run=args.attacks,
                           device=device, verbose=True)

    for images, labels in loader:
        adversary.run_standard_evaluation(images, labels, bs=bs)


def get_model(args, ds):
    '''Given arguments and a dataset object, returns an ImageNet model (with appropriate last layer changes to
    fit the target dataset) and a checkpoint.The checkpoint is set to None if noe resuming training.
    '''
    finetuned_model_path = os.path.join(
        args.out_dir, args.exp_name, 'checkpoint.pt.latest')
    if args.resume or args.eval_only:
        assert os.path.exists(finetuned_model_path), finetuned_model_path
        model, checkpoint = resume_finetuning_from_checkpoint(
            args, ds, finetuned_model_path)
    else:
        if args.dataset in list(transfer_datasets.TRANSFER_DATASETS.keys()) and not args.cifar10_cifar10:
            old_val = ds.num_classes
            ds.num_classes = 1000 #NOTE: to enable loading imagenet weights
            model, _ = make_and_restore_model(
                arch=pytorch_models[args.arch](
                    args.pytorch_pretrained) if args.arch in pytorch_models.keys() else args.arch,
                dataset=ds, resume_path=args.model_path, pytorch_pretrained=args.pytorch_pretrained,
                add_custom_forward=args.arch in pytorch_models.keys())
            ds.num_classes = old_val
            checkpoint = None
        else:
            model, _ = make_and_restore_model(arch=args.arch, dataset=ds, resume_path=args.model_path,
                                              pytorch_pretrained=args.pytorch_pretrained)
            checkpoint = None

        if not args.no_replace_last_layer and not args.eval_only:
            print(f'[Replacing the last layer with {args.additional_hidden} '
                  f'hidden layers and 1 classification layer that fits the {args.dataset} dataset.]')
            while hasattr(model, 'model'):
                model = model.model
            model = fine_tunify.ft(
                args.arch, model, ds.num_classes, args.additional_hidden)
            model, checkpoint = make_and_restore_model(arch=model, dataset=ds,
                                                       add_custom_forward=args.additional_hidden > 0 or args.arch in pytorch_models.keys())
        else:
            print('[NOT replacing the last layer]')
    return model, checkpoint


def get_dataset_and_loaders(args):
    '''Given arguments, returns a datasets object and the train and validation loaders.
    '''
    if args.dataset in ['imagenet', 'stylized_imagenet']:
        ds = datasets.ImageNet(args.data)
        train_loader, validation_loader = ds.make_loaders(
            only_val=args.eval_only, batch_size=args.batch_size, workers=args.workers)
    elif args.cifar10_cifar10:
        ds = datasets.CIFAR(args.data)
        train_loader, validation_loader = ds.make_loaders(
            only_val=args.eval_only, batch_size=args.batch_size, workers=args.workers)
    else:
        ds, (train_loader, validation_loader) = transfer_datasets.make_loaders(
            args.dataset, args.batch_size, args.workers, args.subset)
        if type(ds) == int:
            new_ds = datasets.ImageNet('')
            new_ds.num_classes = ds
            new_ds.mean = ch.tensor([0., 0., 0.])
            new_ds.std = ch.tensor([1., 1., 1.])
            ds = new_ds
    return ds, train_loader, validation_loader


def resume_finetuning_from_checkpoint(args, ds, finetuned_model_path):
    '''Given arguments, dataset object and a finetuned model_path, returns a model
    with loaded weights and returns the checkpoint necessary for resuming training.
    '''
    print('[Resuming finetuning from a checkpoint...]')
    if args.dataset in list(transfer_datasets.TRANSFER_DATASETS.keys()) and not args.cifar10_cifar10:
        model, _ = make_and_restore_model(
            arch=pytorch_models[args.arch](
                args.pytorch_pretrained) if args.arch in pytorch_models.keys() else args.arch,
            dataset=ds, add_custom_forward=args.arch in pytorch_models.keys(), attack_model=args.attack_model)
        while hasattr(model, 'model'):
            model = model.model
        model = fine_tunify.ft(
            args.arch, model, ds.num_classes, args.additional_hidden)
        model, checkpoint = make_and_restore_model(arch=model, dataset=ds, resume_path=finetuned_model_path,
                                                   add_custom_forward=args.additional_hidden > 0 or args.arch in pytorch_models.keys(),
                                                   attack_model=args.attack_model)
    else:
        model, checkpoint = make_and_restore_model(
            arch=args.arch, dataset=ds, resume_path=finetuned_model_path)
    return model, checkpoint


def args_preprocess(args):
    '''
    Fill the args object with reasonable defaults, and also perform a sanity check to make sure no
    args are missing.
    '''
    if args.adv_train and eval(args.eps) == 0:
        print('[Switching to standard training since eps = 0]')
        args.adv_train = 0

    if args.pytorch_pretrained:
        assert not args.model_path, 'You can either specify pytorch_pretrained or model_path, not together.'

    # CIFAR10 to CIFAR10 assertions
    if args.cifar10_cifar10:
        assert args.dataset == 'cifar10'

#    if args.data != '':
#        cs.CALTECH101_PATH = cs.CALTECH256_PATH = cs.PETS_PATH = cs.CARS_PATH = args.data
#        cs.FGVC_PATH = cs.FLOWERS_PATH = cs.DTD_PATH = cs.SUN_PATH = cs.FOOD_PATH = cs.BIRDS_PATH = args.data

    ALL_DS = list(DATASETS.keys()) + list(TRANSFER_DATASETS.keys())
    assert args.dataset in ALL_DS

    # Important for automatic job retries on the cluster in case of premptions. Avoid uuids.
    assert args.exp_name != None

    # Preprocess args
    args = defaults.check_and_fill_args(args, defaults.CONFIG_ARGS, None)
    if not args.eval_only:
        args = defaults.check_and_fill_args(args, defaults.TRAINING_ARGS, None)
    if args.adv_train or args.adv_eval:
        args = defaults.check_and_fill_args(args, defaults.PGD_ARGS, None)
    args = defaults.check_and_fill_args(args, defaults.MODEL_LOADER_ARGS, None)

    return args


if __name__ == "__main__":
    args = parser.parse_args()
    args = args_preprocess(args)

    pytorch_models = {
        'alexnet': models.alexnet,
        'vgg16': models.vgg16,
        'vgg16_bn': models.vgg16_bn,
        'squeezenet': models.squeezenet1_0,
        'densenet': models.densenet161,
        'shufflenet': models.shufflenet_v2_x1_0,
        'mobilenet': models.mobilenet_v2,
        'resnext50_32x4d': models.resnext50_32x4d,
        'mnasnet': models.mnasnet1_0,
    }

    main(args)
