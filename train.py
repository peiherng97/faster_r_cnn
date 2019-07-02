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
from backbone import vgg16
from model import Model
from tensorboardX import SummaryWriter
from extension.lr_scheduler import WarmUpMultiStepLR


device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
def train(dataset, batch_size, backbone_name, config):
    epochs = config.epochs
    dataloader = DataLoader(dataset,batch_size,num_workers=4)   
    backbone  = vgg16(pretrained = True)
    model = Model(backbone = backbone, num_classes = config.num_classes, pooler_mode = 'pooling',
                  anchor_ratios = config.anchor_ratios, anchor_sizes = config.anchor_sizes, 
                  rpn_pre_nms_top_n = config.RPN_PRE_NMS_TOP_N, 
                  rpn_post_nms_top_n = config.RPN_POST_NMS_TOP_N, 
                  anchor_smooth_l1_loss_beta = config.ANCHOR_SMOOTH_L1_LOSS_BETA, 
                  proposal_smooth_l1_loss_beta = config.PROPOSAL_SMOOTH_L1_LOSS_BETA).cuda()
    optimizer = optim.SGD(model.parameters(), lr = config.LEARNING_RATE, 
                          momentum = config.MOMENTUM, 
                          weight_decay = config.STEP_LR_GAMMA,
                          )
    scheduler = WarmUpMultiStepLR(optimizer, milestones=config.STEP_LR_SIZES, gamma=config.STEP_LR_GAMMA,
                              factor=config.WARM_UP_FACTOR, num_iters=config.WARM_UP_NUM_ITERS)
    
    
    iter_ = 0
    for epoch in range(epochs):
        for _, (_, image_batch, _, bboxes_batch, labels_batch) in enumerate(dataloader):
            img = image_batch.to(device)
            bboxes_gt = bboxes_batch.to(device)
            category_label = labels_batch.to(device)
            iter_ += 1
            batch_size = img.shape[0]
            model.zero_grad()
            anchor_objectness_losses, anchor_transformer_losses, proposal_class_losses, proposal_transformer_losses = model(img,bboxes_gt,category_label)
            anchor_objectness_loss = anchor_objectness_losses.mean()
            anchor_transformer_loss = anchor_transformer_losses.mean()
            proposal_class_loss = proposal_class_losses.mean()
            proposal_transformer_loss = proposal_transformer_losses.mean()
            loss = anchor_objectness_loss + anchor_transformer_loss + proposal_class_loss + proposal_transformer_loss
            loss.backward()
            optimizer.step()
            
if __name__ == '__main__':
    config = utils.Params('config.json')
    coco = CocoDetection(root=config.dataset_name, annFile = config.annFile, min_size = config.min_size, max_size = config.max_size)
    train(dataset = coco, batch_size = config.BATCH_SIZE, backbone_name = config.backbone_name, config = config)
    
    
