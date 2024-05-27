from typing import *
import os
import numpy as np

import torch
from torch.utils.data import Dataset, Subset
from torchvision import transforms, datasets

# custom datasets
from dtd import DTD
from aircraft import FGVCAircraft
from caltech import Caltech101, Caltech256

# If you want to read ImageNet data make sure your val directory is preprocessed to look like the train directory,
# e.g. by running this script https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh

DS_LOC = './data'

_TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomSizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

_TEST_TRANSFORM = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])


# functions for loading individual datasets

def _cifar10(split: str) -> Dataset:
    if split == "train":
        is_train = True
        transform = _TRAIN_TRANSFORM
    else:
        is_train = False
        transform = _TEST_TRANSFORM

    ds = datasets.CIFAR10(f"{DS_LOC}/CIFAR10", train=is_train, download=True, transform=transform)
    print(f'Size of {split} set: {ds.__len__()} !!!!!!!!')
    return ds


def _cifar100(split: str) -> Dataset:
    if split == "train":
        is_train = True
        transform = _TRAIN_TRANSFORM
    else:
        is_train = False
        transform = _TEST_TRANSFORM

    ds = datasets.CIFAR100(f"{DS_LOC}/CIFAR100", train=is_train, download=True, transform=transform)
    print(f'Size of {split} set: {ds.__len__()} !!!!!!!!')
    return ds


def _imagenet(split: str) -> Dataset:
    if split == "train":
        subdir = f"{DS_LOC}/imagenet/train"
        transform = _TRAIN_TRANSFORM
    elif split == "test":
        subdir = f"{DS_LOC}/imagenet/val"
        transform = _TEST_TRANSFORM

    full_ds = datasets.ImageFolder(subdir, transform)
    print(f'Size of {split} set: {full_ds.__len__()} !!!!!!!!')
    return full_ds


def _birds(split) -> Dataset:
    if split == "train":
        subdir = f"{DS_LOC}/birdsnap/train"
        transform = _TRAIN_TRANSFORM
    elif split == "test":
        subdir = f"{DS_LOC}/birdsnap/test"
        transform = _TEST_TRANSFORM

    full_ds = datasets.ImageFolder(subdir, transform)
    print(f'Size of {split} set: {full_ds.__len__()} !!!!!!!!')
    return full_ds


def _caltech101(split) -> Dataset:
    ds = Caltech101(DS_LOC)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    NUM_TRAINING_SAMPLES_PER_CLASS = 30

    class_start_idx = [0]+ [i for i in np.arange(1, len(ds)) if ds.y[i]==ds.y[i-1]+1]

    train_indices = sum([np.arange(start_idx,start_idx + NUM_TRAINING_SAMPLES_PER_CLASS).tolist() for start_idx in class_start_idx],[])
    test_indices = list((set(np.arange(1, len(ds))) - set(train_indices) ))

    if split == 'train':
        indices = train_indices
        transform = _TRAIN_TRANSFORM
    else:
        indices = test_indices
        transform = _TEST_TRANSFORM

    split_ds = TransformedDataset(Subset(ds, indices), transform=transform)
    print(f'Size of {split} set: {split_ds.__len__()} !!!!!!!!')
    return split_ds


def _caltech256(split) -> Dataset:
    ds = Caltech256(DS_LOC)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    NUM_TRAINING_SAMPLES_PER_CLASS = 60

    class_start_idx = [0]+ [i for i in np.arange(1, len(ds)) if ds.y[i]==ds.y[i-1]+1]

    train_indices = sum([np.arange(start_idx,start_idx + NUM_TRAINING_SAMPLES_PER_CLASS).tolist() for start_idx in class_start_idx],[])
    test_indices = list((set(np.arange(1, len(ds))) - set(train_indices) ))

    if split == 'train':
        indices = train_indices
        transform = _TRAIN_TRANSFORM
    else:
        indices = test_indices
        transform = _TEST_TRANSFORM

    split_ds = TransformedDataset(Subset(ds, indices), transform=transform)
    print(f'Size of {split} set: {split_ds.__len__()} !!!!!!!!')
    return split_ds


def _dtd(split) -> Dataset:
    if split == 'train':
        is_train = True
        transform = _TRAIN_TRANSFORM
    else:
        is_train = False
        transform = _TEST_TRANSFORM
    ds = DTD(root=f'{DS_LOC}/dtd', train=is_train, transform=transform)
    print(f'Size of {split} set: {ds.__len__()} !!!!!!!!')
    return ds


def _aircraft(split) -> Dataset:
    if split == 'train':
        is_train = True
        transform = _TRAIN_TRANSFORM
    else:
        is_train = False
        transform = _TEST_TRANSFORM
    ds = FGVCAircraft(root=f'{DS_LOC}/fgvc-aircraft-2013b', train=is_train, transform=transform)
    print(f'Size of {split} set: {ds.__len__()} !!!!!!!!')
    return ds


def _pets(split) -> Dataset:
    if split == "train":
        subdir = f"{DS_LOC}/pets/train"
        transform = _TRAIN_TRANSFORM
    elif split == "test":
        subdir = f"{DS_LOC}/pets/test"
        transform = _TEST_TRANSFORM

    full_ds = datasets.ImageFolder(subdir, transform)
    print(f'Size of {split} set: {full_ds.__len__()} !!!!!!!!')
    return full_ds


def _food(split) -> Dataset:
    if split == "train":
        subdir = f"{DS_LOC}/food-101/train"
        transform = _TRAIN_TRANSFORM
    elif split == "test":
        subdir = f"{DS_LOC}/food-101/test"
        transform = _TEST_TRANSFORM

    full_ds = datasets.ImageFolder(subdir, transform)
    print(f'Size of {split} set: {full_ds.__len__()} !!!!!!!!')
    return full_ds


def _flowers(split) -> Dataset:
    if split == "train":
        subdir = f"{DS_LOC}/flowers_new/train"
        transform = _TRAIN_TRANSFORM
    elif split == "test":
        subdir = f"{DS_LOC}/flowers_new/test"
        transform = _TEST_TRANSFORM

    full_ds = datasets.ImageFolder(subdir, transform)
    print(f'Size of {split} set: {full_ds.__len__()} !!!!!!!!')
    return full_ds


def _sun397(split) -> Dataset:
    if split == "train":
        subdir = f"{DS_LOC}/SUN397/splits_01/train"
        transform = _TRAIN_TRANSFORM
    elif split == "test":
        subdir = f"{DS_LOC}/SUN397/splits_01/test"
        transform = _TEST_TRANSFORM

    full_ds = datasets.ImageFolder(subdir, transform)
    print(f'Size of {split} set: {full_ds.__len__()} !!!!!!!!')
    return full_ds


def _cars(split) -> Dataset:
    if split == "train":
        subdir = f"{DS_LOC}/cars_new/train"
        transform = _TRAIN_TRANSFORM
    elif split == "test":
        subdir = f"{DS_LOC}/cars_new/test"
        transform = _TEST_TRANSFORM

    full_ds = datasets.ImageFolder(subdir, transform)
    print(f'Size of {split} set: {full_ds.__len__()} !!!!!!!!')
    return full_ds


# data info dict
_DS_DICT = {
	"imagenet": {
		"ds_fn": _imagenet,
		"num_classes": 1000,
		"mean": [0.485, 0.456, 0.406],
		"stddev": [0.229, 0.224, 0.225]
	},
	"birds": {
		"ds_fn": _birds,
		"num_classes": 500,
		"mean": [0., 0., 0.],
		"stddev": [1., 1., 1.]
	},
	"caltech101": {
		"ds_fn": _caltech101,
		"num_classes": 101,
		"mean": [0., 0., 0.],
		"stddev": [1., 1., 1.]
	},
	"caltech256": {
		"ds_fn": _caltech256,
		"num_classes": 257,
		"mean": [0., 0., 0.],
		"stddev": [1., 1., 1.]
	},
	"cifar10": {
		"ds_fn": _cifar10,
		"num_classes": 10,
		"mean": [0.4914, 0.4822, 0.4465],
		"stddev": [0.2023, 0.1994, 0.2010]
	},
	"cifar100": {
		"ds_fn": _cifar100,
		"num_classes": 100,
		"mean": [0.5071, 0.4867, 0.4408],
		"stddev": [0.2675, 0.2565, 0.2761]
	},
	"dtd": {
		"ds_fn": _dtd,
		"num_classes": 47,
		"mean": [0., 0., 0.],
		"stddev": [1., 1., 1.]
	},
	"aircraft": {
		"ds_fn": _aircraft,
		"num_classes": 100,
		"mean": [0., 0., 0.],
		"stddev": [1., 1., 1.]
	},
	"food": {
		"ds_fn": _food,
		"num_classes": 101,
		"mean": [0., 0., 0.],
		"stddev": [1., 1., 1.]
	},
	"flowers": {
		"ds_fn": _flowers,
		"num_classes": 102,
		"mean": [0., 0., 0.],
		"stddev": [1., 1., 1.]
	},
	"pets": {
		"ds_fn": _pets,
		"num_classes": 37,
		"mean": [0., 0., 0.],
		"stddev": [1., 1., 1.]
	},
	"sun397": {
		"ds_fn": _sun397,
		"num_classes": 397,
		"mean": [0., 0., 0.],
		"stddev": [1., 1., 1.]
	},
	"cars": {
		"ds_fn": _cars,
		"num_classes": 196,
		"mean": [0., 0., 0.],
		"stddev": [1., 1., 1.]
	},
}

DATASETS = list(_DS_DICT.keys())


# other functions
def get_dataset(dataset: str, split: str) -> Dataset:
    """Return the dataset as a PyTorch Dataset object"""
    return _DS_DICT[dataset]['ds_fn'](split)


def get_num_classes(dataset: str):
    """Return the number of classes in the dataset. """
    return _DS_DICT[dataset]['num_classes']


def get_normalize_layer(dataset: str) -> torch.nn.Module:
    """Return the dataset's normalization layer"""
    return NormalizeLayer(_DS_DICT[dataset]['mean'], _DS_DICT[dataset]['stddev'])


class NormalizeLayer(torch.nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset mean
      and dividing by the dataset standard deviation.

      In order to certify radii in original coordinates rather than standardized coordinates, we
      add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
      layer of the classifier rather than as a part of preprocessing as is typical.
      """

    def __init__(self, means: List[float], sds: List[float]):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """
        super(NormalizeLayer, self).__init__()
        self.means = torch.tensor(means).to('cuda:0')
        self.sds = torch.tensor(sds).to('cuda:0')

    def forward(self, input: torch.tensor):
        (batch_size, num_channels, height, width) = input.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        return (input - means) / sds


class TransformedDataset(Dataset):
    def __init__(self, ds, transform=None):
        self.transform = transform
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample, label = self.ds[idx]
        if self.transform:
            sample = self.transform(sample)
            if sample.shape[0] == 1:
                sample = sample.repeat(3,1,1)
        return sample, label