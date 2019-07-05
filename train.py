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
from coco import CocoDetection
from backbone import vgg16
from model import Model
from tensorboardX import SummaryWriter
from extention.lr_scheduler import WarmUpMultiStepLR
from logger import Logger 
#nvidia-smi | grep 'python' | awk '{ print $3 }' | xargs -n1 kill -9

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

def train(dataset, batch_size, backbone_name, config, path_to_checkpoints_dir):
    epochs = config.epochs
    dataloader = DataLoader(dataset,batch_size,num_workers=4)   
    Logger.i('Found {:d} samples' .format(len(dataset)))
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
    step = 0
    steps_to_display = int(len(dataloader)/2)
    time_checkpoint = time.time()
    losses = deque(maxlen=100)
    summary_writer = SummaryWriter(os.path.join(path_to_checkpoints_dir, 'summaries'))
    num_steps_to_finish = epochs * len(dataloader)
    num_steps_to_save = len(dataloader) 
    #num_steps_to_save = len(dataloader)
    for epoch in range(epochs):
        for _, (image_id, image_batch, _, bboxes_batch, labels_batch, skip) in enumerate(dataloader):
            if(skip == 0):
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
                loss.backward()
                optimizer.step()
                scheduler.step()
                losses.append(loss.item())
                summary_writer.add_scalar('train/anchor_objectness_loss', anchor_objectness_loss.item(), step)
                summary_writer.add_scalar('train/anchor_transformer_loss', anchor_transformer_loss.item(), step)
                summary_writer.add_scalar('train/proposal_class_loss', proposal_class_loss.item(), step)
                summary_writer.add_scalar('train/proposal_transformer_loss', proposal_transformer_loss.item(), step)
                summary_writer.add_scalar('train/loss', loss.item(), step)
                step += 1
                if(step % steps_to_display) == 0:
                    elapsed_time = time.time() - time_checkpoint
                    time_checkpoint = time.time()
                    steps_per_sec = steps_to_display / elapsed_time
                    samples_per_sec = batch_size * steps_per_sec
                    eta = (num_steps_to_finish - step) / steps_per_sec / 3600
                    lr = scheduler.get_lr()[0]
                    avg_loss = sum(losses) / len(losses)
                    Logger.i(f'[Step {step}] Avg. Loss = {avg_loss:.6f}, Learning Rate = {lr:.8f} ({samples_per_sec:.2f} samples/sec; ETA {eta:.1f} hrs)')
                if(step % num_steps_to_save == 0 or (step == num_steps_to_finish)):
                    path_to_checkpoint = model.save(path_to_checkpoints_dir, step, optimizer, scheduler)
                    Logger.i(f'Model has been saved to {path_to_checkpoint}')


    Logger.i('Done') 

if __name__ == '__main__':
    config = utils.Params('config.json')
    path_to_output_dir = config.output_dir
    backbone_name = config.backbone_name
    dataset_name  = config.dataset_name
    path_to_checkpoints_dir = os.path.join(path_to_output_dir, 'checkpoints-{:s}-{:s}-{:s}-{:s}'.format(
                               time.strftime('%Y%m%d%H%M%S'), dataset_name, backbone_name, str(uuid.uuid4()).split('-')[0]))
    os.makedirs(path_to_checkpoints_dir)
    Logger.initialize(os.path.join(path_to_checkpoints_dir, 'train.log'))
    coco = CocoDetection(root=config.dataset_name, annFile = config.annFile, min_size = config.min_size, max_size = config.max_size)
    train(dataset = coco, batch_size = config.BATCH_SIZE, backbone_name = config.backbone_name, config = config, path_to_checkpoints_dir = path_to_checkpoints_dir)
    
    
    





