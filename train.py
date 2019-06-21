#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 14:33:58 2019

@author: ph97
"""

import argparse
import logging
import os
import utils
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
from torch.utils.data import DataLoader
from coco import CocoDetection
import backbone.vgg16
#parser = argparse.ArgumentParser()
#parser.add_argument('--data_dir', default='', help="Directory containing the dataset")
#parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
#parser.add_argument('--restore_file', default=None,
#                    help="Optional, name of the file in --model_dir containing weights to reload before \
#                    training") # 'best' or 'train'

def train(dataset, batch_size, backbone_name,):
    dataloader = DataLoader(dataset,batch_size,num_workers=4)   
    backbone_cnn = vgg16(pretrained = True)
if __name__ == '__main__':
    config = utils.Params('config.json')
    coco = CocoDetection(root=config.dataset_name, annFile = config.annFile, min_size = config.min_size, max_size = config.max_size)
    train(dataset = coco, batch_size = config.BATCH_SIZE, backbone_name = config.backbone_name)
    
    
