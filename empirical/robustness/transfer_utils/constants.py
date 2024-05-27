from torchvision import transforms

HOME = "/home/pvaishnavi/data"

FGVC_PATH = f"{HOME}/fgvc-aircraft-2013b/"
CIFAR10_PATH = f"{HOME}/CIFAR10/"

CIFAR100_PATH = f"{HOME}/CIFAR100/"

# Planes dataset

# Oxford Flowers dataset
FLOWERS_PATH = f"{HOME}/flowers_new/"

# DTD dataset
DTD_PATH = f"{HOME}/dtd/"

# Stanford Cars dataset
CARS_PATH = f"{HOME}/cars_new"

# SUN397 dataset
SUN_PATH = f"{HOME}/SUN397/splits_01/"

# FOOD dataset
FOOD_PATH = f"{HOME}/food-101"

# BIRDS dataset
BIRDS_PATH = f"{HOME}/birdsnap"

# PETS dataset
PETS_PATH = f"{HOME}/pets"

# Caltech datasets
CALTECH101_PATH = f"{HOME}/caltech101"
CALTECH256_PATH = f"{HOME}/caltech256"

# Data Augmentation defaults
TRAIN_TRANSFORMS = transforms.Compose([
            # transforms.Resize(32),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

TEST_TRANSFORMS = transforms.Compose([
        # transforms.Resize(32),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
