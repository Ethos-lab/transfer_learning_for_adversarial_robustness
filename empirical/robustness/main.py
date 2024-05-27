"""
The main file, which exposes the robustness command-line tool, detailed in
:doc:`this walkthrough <../example_usage/cli_usage>`.
"""
import shutil
from argparse import ArgumentParser
import os
import sys
import logging
import git
import torch as ch

try:
    from .model_utils import make_and_restore_model
    from .datasets import DATASETS
    from .transfer_datasets import TRANSFER_DATASETS
    from .train_while import train_model, eval_model
    from .tools import constants, helpers
    from .defaults import check_and_fill_args
    from . import defaults, __version__
    from . import datasets
    from . import transfer_datasets
except:
    raise ValueError("Make sure to run with python -m (see README.md)")


parser = ArgumentParser()
parser = defaults.add_args_to_parser(defaults.CONFIG_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.MODEL_LOADER_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.TRAINING_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.PGD_ARGS, parser)


def init_logger(logfilename, resume=False):
    print(f'Creating log file at {logfilename}')
    if os.path.exists(logfilename) and (not resume):
        os.remove(logfilename)

    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler(logfilename))
    logger.addHandler(logging.StreamHandler(sys.stdout))

    return logger


def get_dataset_and_loaders(args):
    '''Given arguments, returns a datasets object and the train and validation loaders.
    '''
    if args.dataset in ['imagenet', 'stylized_imagenet']:
        ds = datasets.ImageNet(args.data)
        train_loader, validation_loader = ds.make_loaders(data_aug=bool(args.data_aug), only_val=args.eval_only,
            batch_size=args.batch_size, workers=args.workers)
    else:
        ds, (train_loader, validation_loader) = transfer_datasets.make_loaders(
            args.dataset, args.batch_size, args.workers, subset=None)
        if type(ds) == int:
            new_ds = datasets.ImageNet('')
            new_ds.num_classes = ds
            new_ds.mean = ch.tensor([0., 0., 0.])
            new_ds.std = ch.tensor([1., 1., 1.])
            ds = new_ds
    return ds, train_loader, validation_loader


def main(args, store=None):
    '''Given arguments from `setup_args` and a store from `setup_store`,
    trains as a model. Check out the argparse object in this file for
    argument options.
    '''
    # Init logger
    save_dir = f'{args.out_dir}/{args.exp_name}'
    if not os.path.exists(save_dir): os.makedirs(save_dir, exist_ok=True)
    logger = init_logger(f'{save_dir}/{"log.txt" if not args.eval_only else "eval.txt"}', args.resume)
    # MAKE DATASET AND LOADERS
    dataset, train_loader, val_loader = get_dataset_and_loaders(args)

    train_loader = helpers.DataPrefetcher(train_loader)
    val_loader = helpers.DataPrefetcher(val_loader)
    loaders = (train_loader, val_loader)

    # MAKE MODEL
    model, checkpoint = make_and_restore_model(arch=args.arch,
            dataset=dataset, resume_path=args.resume)
    if 'module' in dir(model): model = model.module

    logger.info(model)
    logger.info(args)
    logger.info(f'Num classes: {dataset.num_classes}')
    logger.info(f'Mean/Std: {dataset.mean}/{dataset.std}')
    if args.eval_only:
        result_log = eval_model(args, model, val_loader, store=store)
        logger.info(result_log)
        sys.exit()


    if not args.resume_optimizer: checkpoint = None
    model = train_model(args, model, loaders, store=store,
                        checkpoint=checkpoint, logger=logger)
    return model

def setup_args(args):
    '''
    Fill the args object with reasonable defaults from
    :mod:`robustness.defaults`, and also perform a sanity check to make sure no
    args are missing.
    '''
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

    args = setup_args(args)
    final_model = main(args)
