import torch
import numpy as np
import argparse
import os
import pprint

from detectron2.data.datasets import register_coco_instances

from utils import *

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('datapath', type=str, 
        help='dataset path')

    args = parser.parse_args()

    return args

def train():
    args = parse_args()

    # register dataset
    register_coco_instances('deepsports_val', {}, os.path.join(args.datapath, 'val.json'),
                            os.path.join(args.datapath, 'val'))
    register_coco_instances('deepsports_train', {}, os.path.join(args.datapath, 'train.json'),
                            os.path.join(args.datapath, 'train'))
    register_coco_instances("deepsports_test", {}, os.path.join(args.datapath, "test.json"),
                            os.path.join(args.datapath, "test"))
    
    visualize('deepsports_train', 1, True)
    print(pprint.pformat(dataset_analysis('deepsports_train')))
    print(pprint.pformat(dataset_analysis('deepsports_val')))
    print(pprint.pformat(dataset_analysis('deepsports_test')))
    
if __name__ == '__main__':
    train()