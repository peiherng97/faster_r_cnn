#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 13:09:26 2019

@author: ph97
"""
import os
from typing import Union, Tuple, List, Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from bbox import BBox
from extention.functional import beta_smooth_l1_loss
from pooler import Pooler
from region_proposal_network import RegionProposalNetwork
from support.layer.nms import nms


class Model(nn.Module):

    def __init__(self, backbone, num_classes, pooler_mode,
                 anchor_ratios, anchor_sizes,
                 rpn_pre_nms_top_n, rpn_post_nms_top_n,
                 anchor_smooth_l1_loss_beta, proposal_smooth_l1_loss_beta):
        super().__init__()
        self.features, hidden, num_features_out, num_hidden_out = backbone.features()
        self._bn_modules = nn.ModuleList([it for it in self.features.modules() if isinstance(it, nn.BatchNorm2d)] +
                                         [it for it in hidden.modules() if isinstance(it, nn.BatchNorm2d)])
        # NOTE: It's crucial to freeze batch normalization modules for few batches training, which can be done by following processes
        #       (1) Change mode to `eval`
        #       (2) Disable gradient (we move this process into `forward`)
        for bn_module in self._bn_modules:
            for parameter in bn_module.parameters():
                parameter.requires_grad = False

        self.rpn = RegionProposalNetwork(num_features_out, anchor_ratios, anchor_sizes, rpn_pre_nms_top_n, rpn_post_nms_top_n, anchor_smooth_l1_loss_beta)
        self.detection = Model.Detection(pooler_mode, hidden, num_hidden_out, num_classes, proposal_smooth_l1_loss_beta)

    def forward(self, image_batch, gt_bboxes_batch, gt_classes_batch):
        # disable gradient for each forwarding process just in case model was switched to `train` mode at any time
        for bn_module in self._bn_modules:
            bn_module.eval()
        '''
        Training mode
        Generate anchor box 
        Pass through RPN proposals and get anchor objectness score, anchor deltas and losses
        Pass through detection model to get class prediction, proposal deltas and losses
           
        Testing mode
        Addition of detection generation module and removal 
        '''
        features = self.features(image_batch)
        batch_size, _, image_height, image_width = image_batch.shape
        _, _, features_height, features_width = features.shape
        anchor_bboxes = self.rpn.generate_anchors(image_width, image_height, num_x_anchors=features_width, num_y_anchors=features_height).to(features).repeat(batch_size, 1, 1)
        
        if self.training:
            anchor_objectnesses, anchor_transformers, anchor_objectness_losses, anchor_transformer_losses = self.rpn.forward(features, anchor_bboxes, gt_bboxes_batch, image_width, image_height)
            proposal_bboxes = self.rpn.generate_proposals(anchor_bboxes, anchor_objectnesses, anchor_transformers, image_width, image_height).detach()  # it's necessary to detach `proposal_bboxes` here
            proposal_classes, proposal_transformers, proposal_class_losses, proposal_transformer_losses = self.detection.forward(features, proposal_bboxes, gt_classes_batch, gt_bboxes_batch)
            return anchor_objectness_losses, anchor_transformer_losses, proposal_class_losses, proposal_transformer_losses
        else:
            anchor_objectnesses, anchor_transformers = self.rpn.forward(features)
            proposal_bboxes = self.rpn.generate_proposals(anchor_bboxes, anchor_objectnesses, anchor_transformers, image_width, image_height)
            proposal_classes, proposal_transformers = self.detection.forward(features, proposal_bboxes)
            detection_bboxes, detection_classes, detection_probs, detection_batch_indices = self.detection.generate_detections(proposal_bboxes, proposal_classes, proposal_transformers, image_width, image_height)
            return detection_bboxes, detection_classes, detection_probs, detection_batch_indices

    def save(self, path_to_checkpoints_dir: str, step: int, optimizer: Optimizer, scheduler: _LRScheduler) -> str:
        path_to_checkpoint = os.path.join(path_to_checkpoints_dir, f'model-{step}.pth')
        checkpoint = {
            'state_dict': self.state_dict(),
            'step': step,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }
        torch.save(checkpoint, path_to_checkpoint)
        return path_to_checkpoint

    def load(self, path_to_checkpoint: str, optimizer: Optimizer = None, scheduler: _LRScheduler = None) -> 'Model':
        checkpoint = torch.load(path_to_checkpoint)
        self.load_state_dict(checkpoint['state_dict'])
        step = checkpoint['step']
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return step
    
    class Detection(nn.Module):
        def __init__(self, pooler, hidden, num_hidden_out, num_classes, proposal_smooth_l1_loss_beta):
            super().__init__() 
            self.num_classes = num_classes
            self._proposal_class = nn.Linear(num_hidden_out, num_classes)
            self._proposal_transformer = nn.Linear(num_hidden_out, num_classes*4)
            self._proposal_smooth_l1_loss_beta = proposal_smooth_l1_loss_beta
            self._transformer_normalize_mean = torch.tensor([0,0,0,0],dtype=torch.float)
            self._transformer_normalize_std = torch.tensor([0.1,0.1,0.2,0.2],dtype=torch.float)
            self.hidden = hidden
            self._pooler_mode = pooler 
            
        def forward(self, features, proposal_bboxes, gt_classes_batch, gt_bboxes_batch):
            batch_size = features.shape[0]
            '''assign -1 to all labels first (Not sure why for now)
               calculate iou for proposals with ground truth bboxes
               proposal_assignments contains the highest ranked ground truth bbox for each proposal
               get the maximum iou of each proposal with respect to each ground truth box and store highest ground truth box in proposal assigments
               assign 0 to all labels with iou less than 0.5
               assign class labels to all proposals with iou higher than 0.5 
               foreground ( >= 0.5 ) background ( < 0.5 )
               Take total of 128 proposal samples
               split fg and bg into 0.25 : 0.75 ratio after reshuffling
            '''
            labels = torch.full((batch_size,proposal_bboxes.shape[1]), -1, dtype=torch.long, device= proposal_bboxes.device)
            ious = BBox.iou(proposal_bboxes,gt_bboxes_batch)
            proposal_max_ious, proposal_assignments = ious.max(dim=2)
            labels[proposal_max_ious < 0.5] = 0
            fg_masks = proposal_max_ious >= 0.5
            if len(fg_masks.nonzero()) > 0:
                labels[fg_masks] = gt_classes_batch[fg_masks.nonzero()[:,0],proposal_assignments[fg_masks]]
            fg_indices = (labels > 0).nonzero()
            bg_indices = (labels == 0).nonzero()
            fg_indices = fg_indices[torch.randperm(len(fg_indices))[:min(len(fg_indices),32*batch_size)]]
            bg_indices = bg_indices[torch.randperm(len(bg_indices))[:128 * batch_size - len(fg_indices)]]
            selected_indices = torch.cat([fg_indices,bg_indices],dim=0)
            ''' selected_indices shape : torch.Size([128, 2]) '''
            selected_indices = selected_indices[torch.randperm(len(selected_indices))].unbind(dim=1)
            
            '''
            len(selected_indices) = 128
            Assign ground truth targets of selected indices
            gt_bboxes are formed by 
            Apply ROI pooling on the features with proposal_bboxes generated
            Pass it through a hidden layer, pool and flatten
            '''
            proposal_bboxes = proposal_bboxes[selected_indices]
            gt_bboxes = gt_bboxes_batch[selected_indices[0],proposal_assignments[selected_indices]]
            gt_proposal_classes = labels[selected_indices]
            gt_proposal_transformers = BBox.calc_transformer(proposal_bboxes,gt_bboxes)
            batch_indices = selected_indices[0]
            pool = Pooler.apply(features, proposal_bboxes, proposal_batch_indices = batch_indices, mode = self._pooler_mode)
            pool = pool.view(pool.shape[0],-1)
            hidden = self.hidden(pool)
#            hidden = F.adaptive_max_pool2d(input=hidden,output_size=1)
#            hidden = hidden.view(hidden.shape[0],-1)
            proposal_classes = self._proposal_class(hidden)
            proposal_transformers = self._proposal_transformer(hidden)
            proposal_class_losses, proposal_transformer_losses = self.loss(proposal_classes,proposal_transformers,gt_proposal_classes,gt_proposal_transformers,batch_size,batch_indices)
            return proposal_classes,proposal_transformers, proposal_class_losses, proposal_transformer_losses
        
        def loss(self,proposal_classes,proposal_transformers,gt_proposal_classes,gt_proposal_transformers,batch_size,batch_indices):
            '''
            Only take the loss from the correct class from the bounding box regressor
            scale up target to make regressor easier to learn
            '''
            proposal_transformers = proposal_transformers.view(-1,self.num_classes,4)[torch.arange(len(proposal_transformers)),gt_proposal_classes]
            transformer_normalize_mean = self._transformer_normalize_mean.to(device=gt_proposal_transformers.device)
            transformer_normalize_std = self._transformer_normalize_std.to(device=gt_proposal_transformers.device)
            gt_proposal_transformers = (gt_proposal_transformers - transformer_normalize_mean) / transformer_normalize_std  
            
            cross_entropies = torch.empty(batch_size, dtype=torch.float, device=proposal_classes.device)
            smooth_l1_losses = torch.empty(batch_size, dtype=torch.float, device=proposal_transformers.device)
            
            for batch_index in range(batch_size):
                selected_indices = (batch_indices == batch_index).nonzero().view(-1)
                cross_entropy = F.cross_entropy(input=proposal_classes[selected_indices],target=gt_proposal_classes[selected_indices])
                fg_indices = gt_proposal_classes[selected_indices].nonzero().view(-1)
                smooth_l1_loss = beta_smooth_l1_loss(input=proposal_transformers[selected_indices][fg_indices],target=gt_proposal_transformers[selected_indices][fg_indices],beta=self._proposal_smooth_l1_loss_beta)
                cross_entropies[batch_index] = cross_entropy
                smooth_l1_losses[batch_index] = smooth_l1_loss
                
            return cross_entropies, smooth_l1_losses
        
        def generate_detections(self,proposal_bboxes,proposal_classes,proposal_transformers,image_width,image_height):
            
            '''
            Get proposal deltas for each different class
            Denormalized the deltas
            Duplicate the proposal bboxes for each class
            Apply delta transform on the proposal bboxes for each class
            CLip detection bboxes so the it wont go out of bounds
            
            '''
            batch_size = proposal_bboxes.shape[0]
            
            proposal_transformers = proposal_transformers.view(batch_size,-1,self.num_classes,4)
            transformer_normalize_std = self._transformer_normalize_std.to(device=proposal_transformers.device)
            transformer_normalize_mean = self._transformer_normalize_mean.to(device=proposal_transformers.device)
            proposal_transformers = proposal_transformers * transformer_normalize_std + transformer_normalize_mean
            
            proposal_bboxes = proposal_bboxes.unsqueeze(dim=2).repeat(1,1,self.num_classes,1)
            detection_bboxes = BBox.apply_transformer(proposal_bboxes,proposal_transformers)
            detection_bboxes = BBox.clip(detection_bboxes, left=0, top=0, right=image_width, bottom=image_height)
            detection_probs = F.softmax(proposal_classes, dim=-1)
            
            detection_bboxes_list = []
            detection_classes_list = []
            detection_probs_list = []
            detection_batch_indices_list = []
            
            '''Class iteration starts from 1, ignore background class (0) '''
            for batch_index in range(batch_size):
                for class_ in range(1, self.num_classes):
                    class_bboxes = detection_bboxes[batch_index,:,class_,:]
                    class_probs = detection_probs[batch_index,:,class_]
                    threshold = 0.3
                    kept_indices = nms(class_bboxes, class_probs, threshold)
                    class_bboxes = class_bboxes[kept_indices]
                    class_probs = class_probs[kept_indices]
                    
                    detection_bboxes_list.append(class_bboxes)
                    detection_classes_list.append(torch.full((len(kept_indices),), class_, dtype=torch.int))
                    detection_probs_list.append(class_probs)
                    detection_batch_indices_list.append(torch.full((len(kept_indices),),batch_index,dtype=torch.long))
            
            detection_bboxes_list = torch.cat(detection_bboxes_list,dim=0)
            detection_classes_list = torch.cat(detection_classes_list,dim=0)
            detection_probs_list = torch.cat(detection_probs_list,dim=0)
            detection_batch_indices_list = torch.cat(detection_batch_indices_list,dim=0)
            
            return detection_bboxes_list,detection_classes_list,detection_probs_list,detection_batch_indices_list

                    
                    
            
