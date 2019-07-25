#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 14:33:58 2019

@author: ph97
"""

import argparse
import os
import utils
import time
import uuid
from collections import deque
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
from torch.utils.data import DataLoader
from coco_animal import CocoDetection
from backbone import Resnet101 as resnet101
from model_resnet101 import Model
from tensorboardX import SummaryWriter
from extention.lr_scheduler import WarmUpMultiStepLR
from logger import Logger 
from torch.utils.data import SubsetRandomSampler
#nvidia-smi | grep 'python' | awk '{ print $3 }' | xargs -n1 kill -9

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
print(device)
def train(dataset, batch_size, backbone_name, config):
    summary_writer = SummaryWriter(os.path.join(config.output_dir, 'summaries_r3'))
    epochs = config.epochs
    #train_indices = torch.arange(200)
    #'''sampler=SubsetRandomSampler(train_indices)'''
    dataloader = DataLoader(dataset,batch_size,num_workers=4,shuffle=False,collate_fn = dataset.padding_collate_fn) 
    print('Found {:d} samples' .format(len(dataset)))
    backbone  = resnet101(pretrained = True)
    model = Model(backbone = backbone, num_classes = config.num_classes, pooler_mode = 'align',
                  anchor_ratios = config.anchor_ratios, anchor_sizes = config.anchor_sizes, 
                  rpn_pre_nms_top_n = config.RPN_PRE_NMS_TOP_N, 
                  rpn_post_nms_top_n = config.RPN_POST_NMS_TOP_N, 
                  anchor_smooth_l1_loss_beta = config.ANCHOR_SMOOTH_L1_LOSS_BETA, 
                  proposal_smooth_l1_loss_beta = config.PROPOSAL_SMOOTH_L1_LOSS_BETA).cuda()
    optimizer = optim.SGD(model.parameters(), lr = config.LEARNING_RATE, 
                          momentum = config.MOMENTUM, 
                          weight_decay = config.WEIGHT_DECAY,
                          )
    step = 0
    for epoch in range(epochs):
        for it, (image_id, image_batch, scale, bboxes_batch, labels_batch) in enumerate(dataloader):
#            print('train:',image_id,scale,labels_batch)
                img = image_batch.to(device)
                bboxes_gt = bboxes_batch.to(device)
                category_label = labels_batch.to(device)
                batch_size = img.shape[0]
                model.zero_grad()
                anchor_objectness_losses, anchor_transformer_losses, proposal_class_losses, proposal_transformer_losses = model(img,bboxes_gt,category_label)
                anchor_objectness_loss = anchor_objectness_losses.mean()
                anchor_transformer_loss = anchor_transformer_losses.mean()
                proposal_class_loss = proposal_class_losses.mean()
                proposal_transformer_loss = proposal_transformer_losses.mean()
                loss = anchor_objectness_loss + anchor_transformer_loss + proposal_class_loss + proposal_transformer_loss
                if(it%100 == 0):
                  print(epoch)
                  print(step)
                  print('anchor_o_loss ',anchor_objectness_loss.item())
                  print('anchor_t_loss ',anchor_transformer_loss.item())
                  print('proposal_class_loss ',proposal_class_loss.item())
                  print('proposal_transformer_loss ',proposal_transformer_loss.item())
                  print('total_loss ',loss.item())
                  summary_writer.add_scalar('train/anchor_objectness_loss', anchor_objectness_loss.item(), step)
                  summary_writer.add_scalar('train/anchor_transformer_loss', anchor_transformer_loss.item(), step)
                  summary_writer.add_scalar('train/proposal_class_loss', proposal_class_loss.item(), step)
                  summary_writer.add_scalar('train/proposal_transformer_loss', proposal_transformer_loss.item(), step)
                  summary_writer.add_scalar('train/loss', loss.item(), step)

                if(it%500 == 0):
                  torch.save(model.state_dict(),'outputs/model_resnet101_r3.pth')
                loss.backward()
                optimizer.step()
                step += 1

if __name__ == '__main__':
    config = utils.Params('config_resnet101.json')
    path_to_output_dir = config.output_dir
    backbone_name = config.backbone_name
    dataset_name  = config.dataset_name
    coco = CocoDetection(root=config.dataset_name, annFile = config.annFile, min_size = config.min_size, max_size = config.max_size)
    train(dataset = coco, batch_size = config.BATCH_SIZE, backbone_name = config.backbone_name, config = config)
    
    
    





