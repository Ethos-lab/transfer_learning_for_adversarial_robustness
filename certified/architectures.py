# this file is based on code publicly available at
#   https://github.com/locuslab/smoothing
# written by Jeremy Cohen.

import torch
from archs.resnet import resnet50
import torch.backends.cudnn as cudnn
from transfer_datasets import get_normalize_layer, get_num_classes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_architecture(arch: str, dataset: str, return_latent: bool) -> torch.nn.Module:
    """ Return a neural network (with random weights)

    :param arch: the architecture - should be in the ARCHITECTURES list above
    :param dataset: the dataset - should be in the datasets.DATASETS list
    :return: a Pytorch module
    """
    if arch == "resnet50":
        model = resnet50(pretrained=False, num_classes=get_num_classes(dataset), return_latent=return_latent)
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
    else:
        raise NotImplementedError

    normalize_layer = get_normalize_layer(dataset)
    return torch.nn.Sequential(normalize_layer, model).to(device)
