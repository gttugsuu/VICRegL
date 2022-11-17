# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from pathlib import Path
import argparse
import json
import os
import random
import signal
import sys
import time
import urllib

from tqdm.auto import tqdm
from PIL import Image
import numpy as np
import pandas as pd
from torch import nn, optim
from torch.utils.data import Dataset
from torchvision import transforms
import torch


def get_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate a pretrained model on ImageNet"
    )

    # Data
    parser.add_argument("--data-dir", type=Path, help="path to dataset")
    parser.add_argument(
        "--train-percent",
        default=100,
        type=int,
        choices=(100, 10, 1),
        help="size of traing set in percent",
    )

    # Checkpoint
    parser.add_argument("--pretrained", type=Path, default="../output/bbbc021_from_server/model.pth", help="path to pretrained model")
    parser.add_argument(
        "--exp-dir",
        default="./checkpoint/lincls/",
        type=Path,
        metavar="DIR",
        help="path to checkpoint directory",
    )
    parser.add_argument(
        "--print-freq", default=100, type=int, metavar="N", help="print frequency"
    )
    parser.add_argument("--representation_dir", type=Path, default="../output/bbbc021_from_server/representations_test/", help="Path to save representations")

    # Model
    parser.add_argument("--arch", type=str, default="convnext_small")

    # Optim
    parser.add_argument(
        "--epochs",
        default=100,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--batch-size", default=64, type=int, metavar="N", help="mini-batch size"
    )
    parser.add_argument(
        "--lr-backbone",
        default=0.0,
        type=float,
        metavar="LR",
        help="backbone base learning rate",
    )
    parser.add_argument(
        "--lr-head",
        default=0.3,
        type=float,
        metavar="LR",
        help="classifier base learning rate",
    )
    parser.add_argument(
        "--weight-decay", default=1e-6, type=float, metavar="W", help="weight decay"
    )
    parser.add_argument(
        "--weights",
        default="freeze",
        type=str,
        choices=("finetune", "freeze"),
        help="finetune or freeze resnet weights",
    )

    # Running
    parser.add_argument(
        "--workers",
        default=0,
        type=int,
        metavar="N",
        help="number of data loader workers",
    )

    return parser

class bbbc021_splitbymoa_cell_test(Dataset):
    def __init__(self, transform, cellsize=128, mode='test', train=True):
        self.mode = mode
        if self.mode == 'train':
            if train:
                self.metadata = pd.read_csv('/home/gantugs/working/dino_things/dino_vgg_other/split_by_moa/meta_train.csv')
            else:
                self.metadata = pd.read_csv('/home/gantugs/working/dino_things/dino_vgg_other/split_by_moa/meta_valid.csv')
        elif self.mode == 'test':
            self.metadata = pd.read_csv('/home/gantugs/working/dino_things/dino_vgg_other/split_by_moa/meta_test.csv')
         
        self.transform = transform

    def __len__(self):
        return self.metadata.shape[0]

    def __getitem__(self, i):
        cellimagepath = self.metadata.cellimagepaths[i]
        cellimage = Image.open(cellimagepath).convert('RGB')
        cellimage = self.transform(cellimage)
        return cellimage, Path(cellimagepath).stem

def main():
    
    parser = get_arguments()
    args = parser.parse_args()
    
    gpu = "cuda:0"
    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True

    if "convnext" in args.arch:
        import convnext
        backbone, embedding = convnext.__dict__[args.arch](
            drop_path_rate=0.1,
            layer_scale_init_value=0.0,
        )
    elif "resnet" in args.arch:
        import resnet
        backbone, embedding = resnet.__dict__[args.arch](
            zero_init_residual=True
        )

    state_dict = torch.load(args.pretrained, map_location="cpu")
    if "model" in state_dict:
        state_dict = state_dict["model"]
        state_dict = {
            key.replace("module.backbone.", ""): value
            for (key, value) in state_dict.items()
        }
    backbone.load_state_dict(state_dict, strict=False)
    
    model = ModelWrapper_nohead(backbone)
    model.cuda(gpu)
    
    backbone.requires_grad_(False)
    
    # data loading code
    transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.168, 0.137, 0.096), (0.175, 0.159, 0.168)), # bbbc021 mean, std
            ]
        )
    
    test_dataset = bbbc021_splitbymoa_cell_test(train=True, mode='test', transform=transform)
    kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, **kwargs)
    
    start_time = time.time()
    model.eval()
    
    args.representation_dir.mkdir(parents=True, exist_ok=True)
        
    for i, (images, names) in enumerate(tqdm(test_loader)):
        # dryrun = True
        # if dryrun == True:
        #     a=1
        #     continue
        representations = model(images.cuda(gpu, non_blocking=True))
        representations = representations.detach().cpu().squeeze().numpy()
        
        # save 
        # for representation, name in zip(representations, names):
        #     representation = representation.detach().cpu().squeeze().numpy()
        #     np.save(f'{args.representation_dir}/{name}', representation)
        np.save(f'{args.representation_dir}/batch_{i}', representations)
        with open(f'{args.representation_dir}/batch_{i}.txt', 'w') as f:
            for name in names:
                f.write(name + '\n')
    
    print("End of evaluation")
             
class ModelWrapper_nohead(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, input_):
        _ , x = self.backbone(input_)
        return x
    
if __name__ == "__main__":
    main()