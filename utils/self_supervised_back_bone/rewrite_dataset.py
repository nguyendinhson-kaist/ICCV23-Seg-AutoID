from torch.utils.data import Dataset
import os
from PIL import Image

import os
import PIL

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

class CustomDataset(Dataset):
    def __init__(
            self,
            root,
            transform=None) -> None:
        super(CustomDataset, self).__init__()
        self.transform = transform
        files = os.listdir(root)
        self.data = [os.path.join(root, file) for file in files]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index) -> tuple:
        file = self.data[index]
        img = Image.open(file)
        if self.transform:
            img = self.transform(img)
        return img, 'class'

def build_dataset(is_train, args):
    """
    Build train/val dataset for finetuning classification
    """
    transform = build_transform(is_train, args)

    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)

    print(dataset)
    return dataset


def build_transform(is_train, args):
    #TODO: should change to MEAN, STD of DATASET
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob, # random erase prob
            re_mode=args.remode, # random erase mode
            re_count=args.recount, # random erase count
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
