import pdb
import os
import sys
import shutil
import time
import logging
import numpy as np
import tqdm

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from architectures import get_architecture
from transfer_datasets import get_dataset, get_num_classes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# for freezing
IDX_TO_LAYER = {
    'resnet50': {
        1: ['conv1', 'bn1'],
        2: ['conv1', 'bn1', 'layer1'],
        3: ['conv1', 'bn1', 'layer1', 'layer2'],
        4: ['conv1', 'bn1', 'layer1', 'layer2', 'layer3'],
        5: ['conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4'],
    }
}
# for appending new final layer
FEAT_DIMS = {
    'resnet50': 2048
}


def init_logger(logfilename, resume=False):
    if os.path.exists(logfilename) and (not resume):
        os.remove(logfilename)

    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler(logfilename))
    logger.addHandler(logging.StreamHandler(sys.stdout))

    return logger


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def requires_grad_(model:torch.nn.Module, requires_grad:bool) -> None:
    for param in model.parameters():
        param.requires_grad_(requires_grad)


def copy_code(outdir):
    """Copies files to the outdir to store complete script with each experiment"""
    # embed()
    code = []
    exclude = set([])
    for root, _, files in os.walk("./code", topdown=True):
        for f in files:
            if not f.endswith('.py'):
                continue
            code += [(root,f)]

    for r, f in code:
        codedir = os.path.join(outdir,r)
        if not os.path.exists(codedir):
            os.mkdir(codedir)
        shutil.copy2(os.path.join(r,f), os.path.join(codedir,f))
    print("Code copied to '{}'".format(outdir))


def prepare_source_model(model, arch, load_path, dataset, logger):
    # prepare pre-trained weights
    assert os.path.exists(load_path)
    checkpoint = torch.load(load_path)
    try:
        sd = checkpoint['state_dict']
    except KeyError:
        sd = checkpoint

    #NOTE: loading on model[1] because get_arch returns model wrapped in nn.DataParallel
    model[1].module.fc = None
    sd = {k.replace('1.module.', ''): v for k, v in sd.items()}
    model[1].module.load_state_dict(sd)
    # update final layer
    model[1].module.fc = nn.Linear(FEAT_DIMS[arch], get_num_classes(dataset)).to(device)

    epoch = checkpoint['epoch'] if 'epoch' in checkpoint.keys() else -1
    logger.info(f"Reusing features from: {load_path} (epoch: {epoch})")

    return model


def freeze_layers(model, arch, freeze, logger):
    #TODO: add support for freezing layers of other models
    if arch != 'resnet50': raise NotImplementedError
    params = []
    list_of_freezing_layer = set()
    list_of_training_layer = set()
    for param_n, param in model[1].module.named_parameters():
        layer_n = param_n.split('.')[0]
        if layer_n in IDX_TO_LAYER[arch][freeze]:
            param.requires_grad = False
            list_of_freezing_layer.add(layer_n)
        else:
            param.requires_grad = True
            list_of_training_layer.add(layer_n)
            params.append(param)

    logger.info(f'Freezing: {list_of_freezing_layer}')
    logger.info(f'Training: {list_of_training_layer}')

    return model, params


def resume_training(model, optimizer, model_path, logger):
    if os.path.isfile(model_path):
        logger.info("=> resuming checkpoint '{}'".format(model_path))
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        starting_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> resume epoch {}".format(checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(model_path))

    return model, optimizer, starting_epoch


def prepare_model(args, logger):
    # init model
    model = get_architecture(args.arch, args.dataset, args.return_latent)

    # load source task weights (if specified)
    if args.pretrained_model != '':
        model = prepare_source_model(model, args.arch, args.pretrained_model, args.dataset, logger)
    else:
        logger.info('Training from scratch')

    # freeze layers as specified
    logger.info(f'Freezing {args.freeze} layers !!!')
    if args.freeze > 0:
        model, train_params = freeze_layers(model, args.arch, args.freeze, logger)
    else:
        train_params = model.parameters()

    criterion = CrossEntropyLoss().to(device)
    optimizer = SGD(train_params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.gamma)

    # resume training if specified
    starting_epoch = 0
    model_path = os.path.join(args.outdir, 'checkpoint.pth.tar')
    if args.resume:
        model, optimizer, starting_epoch = resume_training(model, optimizer, model_path, logger)

    print(model)
    return model, criterion, optimizer, scheduler, model_path, starting_epoch


def prologue(args):
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir, exist_ok=True)

    # Copies files to the outdir to store complete script with each experiment
    copy_code(args.outdir)
    logfilename = os.path.join(args.outdir, 'log.txt')
    logger = init_logger(logfilename, args.resume)
    writer = SummaryWriter(args.outdir)

    train_dataset = get_dataset(args.dataset, 'train')
    test_dataset = get_dataset(args.dataset, 'test')
    pin_memory = (args.dataset == "imagenet")
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch,
                              num_workers=args.workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch,
                             num_workers=args.workers, pin_memory=pin_memory)

    model, criterion, optimizer, scheduler, model_path, starting_epoch =\
        prepare_model(args, logger)

    return train_loader, test_loader, criterion, model, optimizer, scheduler,\
           starting_epoch, logger, model_path, writer


def test(loader, model, criterion, epoch, noise_sd, device, writer=None, noise_type='gaussian'):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    # switch to eval mode
    model.eval()
    pbar = tqdm.tqdm(total=len(loader), leave=False)
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader):
            # measure data loading time
            data_time.update(time.time() - end)

            inputs, targets = inputs.to(device), targets.to(device)

            # augment inputs with noise
            if noise_type == 'gaussian':
                inputs = inputs + torch.randn_like(inputs, device=device) * noise_sd
            elif noise_type == 'uniform':
                inputs =\
                  inputs + torch.rand_like(inputs, device=device) * noise_sd
            else:
                raise ValueError

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1.item(), inputs.size(0))
            top5.update(acc5.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            pbar.update(1)

        if writer:
            writer.add_scalar('loss/test', losses.avg, epoch)
            writer.add_scalar('accuracy/test@1', top1.avg, epoch)
            writer.add_scalar('accuracy/test@5', top5.avg, epoch)
        pbar.close()

        return (losses.avg, top1.avg)



def normalize(x, eps=1e-8):
    return x / (x.norm(dim=1, keepdim=True) + eps)


def check_spectral_norm(m, name='weight'):
    from torch.nn.utils.spectral_norm import SpectralNorm
    for k, hook in m._forward_pre_hooks.items():
        if isinstance(hook, SpectralNorm) and hook.name == name:
            return True
    return False


def apply_spectral_norm(m):
    from torch.nn.utils import spectral_norm
    for layer in m.modules():
        if isinstance(layer, nn.Conv2d):
            spectral_norm(layer)
        elif isinstance(layer, nn.Linear):
            spectral_norm(layer)
        elif isinstance(layer, nn.Embedding):
            spectral_norm(layer)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
