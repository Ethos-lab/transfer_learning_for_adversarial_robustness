from torch.utils.data.dataset import Dataset
from PIL import Image


class DTD(Dataset):
    def __init__(self, root, split="1", train=False, transform=None):
        super().__init__()
        train_path = f"{root}/labels/train{split}.txt"
        val_path = f"{root}/labels/val{split}.txt"
        test_path = f"{root}/labels/test{split}.txt"
        if train:
            self.ims = open(train_path).readlines() + open(val_path).readlines()
        else:
            self.ims = open(test_path).readlines()
        
        self.full_ims = [f"{root}/images/{x}" for x in self.ims]
        
        pth = f"{root}/labels/classes.txt"
        self.c_to_t = {x.strip(): i for i, x in enumerate(open(pth).readlines())}

        self.transform = transform
        self.labels = [self.c_to_t[x.split("/")[0]] for x in self.ims]

    def __getitem__(self, index):
        im = Image.open(self.full_ims[index].strip())
        if self.transform:
            im = self.transform(im)
        return im, self.labels[index]

    def __len__(self):
        return len(self.ims)
