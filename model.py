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

from backbone.base import Base as BackboneBase
from bbox import BBox
from extention.functional import beta_smooth_l1_loss
from roi.pooler import Pooler
from rpn.region_proposal_network import RegionProposalNetwork
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

    def forward(self, image_batch: Tensor,
                gt_bboxes_batch: Tensor = None, gt_classes_batch: Tensor = None) -> Union[Tuple[Tensor, Tensor, Tensor, Tensor],
                                                                                          Tuple[Tensor, Tensor, Tensor, Tensor]]:
        # disable gradient for each forwarding process just in case model was switched to `train` mode at any time
        for bn_module in self._bn_modules:
            bn_module.eval()

        features = self.features(image_batch)

        batch_size, _, image_height, image_width = image_batch.shape
        _, _, features_height, features_width = features.shape
        print(features.shape)
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
        def __init__(self, hidden_layer, num_hidden_out, num_classes, proposal_smooth_l1_loss_beta):
            super().__init__()
            self._proposal_class = nn.Linear(num_hidden_out, num_classes)
            self._proposal_transformer = nn.Linear(num_hidden_out, num_classes*4)
            self._proposal_smooth_l1_loss_beta = proposal_smooth_l1_loss_beta
            self._transformer_normalize_mean = torch.tensor([0,0,0,0],dtype=torch.float)
            self._transformer_normalize_std = torch.tensor([0.1,0.1,0.2,0.2],dtype=torch.float)
            self.hidden_layer = hidden_layer
            
        def forward(self, features, proposal_bboxes, gt_classes_batch, gt_bboxes_batch):
            batch_size = features.shape[0]
            labels = torch.full((batch_size,proposal_bboxes.shape[1]), -1, dtype=torch.long, device= proposal_bboxes.device)
            ious = BBox.iou(proposal_bboxes,gt_bboxes_batch)
            proposal_max_ious, proposal_assignments = ious.max(dim=2)
            labels[proposal_max_ious < 0.5] = 0
            fg_masks = proposal_max_ious >= 0.5
            if len(fg_masks.nonzero()) > 0:
                labels[fg_masks] = gt_classes_batch[fg_masks.nonzero()[:,0],proposal_assignments[fg_masks]]
            fg_indices = (labels > 0).nonzero()
            bg_indices = (labels == 0).nonzero()
            '''Permutation on the indices to randomize'''
            fg_indices = fg_indices[torch.randperm(len(fg_indices))[:min(len(fg_indices),32*batch_size)]]
            bg_indices = bg_indices[torch.randperm(len(bg_indices))[:128 * batch_size - len(fg_indices)]]
            selected_indices = torch.cat([fg_indices,bg_indices],dim=0)
            selected_indices = selected_indices[torch.randperm(len(selected_indices))].unbind(dim=1)
            
            proposal_bboxes = proposal_bboxes[selected_indices]
            gt_bboxes = gt_bboxes_batch[selected_indices[0],proposal_assignments[selected_indices]]
            gt_proposal_classes = labels[selected_indices]
            gt_proposal_transformers = BBox.calc_transformer(proposal_bboxes,gt_bboxes)
            batch_indices = selected_indices[0]

            pool = Pooler.apply(features,proposal_bboxes,proposal_batch_indices,mode=self._pooler_mode)
            hidden = self.hidden(pool)
            hidden = F.adaptive_max_pool2d(input=hidden,output_size=1)
            hidden = hidden.view(hidden.shape[0],-1)
            
            proposal_classes = self._proposal_class(hidden)
            proposal_transformers = self._proposal_transformer(hidden)
            proposal_class_losses, proposal_transformer_losses = self.loss(proposal_classes,proposal_transformers,gt_proposal_classes,gt_proposal_transformers,batch_size,batch_indices)
            
            return proposal_classes,proposal_transformers, proposal_class_losses, proposal_transformer_losses
        
        def loss(self,propsoal_classes,proposal_transformers,gt_proposal_classes,gt_proposal_transformers):
            proposal_transformers = proposal_transformers.view(-1,self.num_classes,4)[]