#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
This code is adapted from https://github.com/potterhsu/easy-faster-rcnn.pytorch
'''
import torch
from torch import Tensor

class BBox(object):
    def __init__(self,left,top,right,bottom):
        super(BBox).__init__()
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom

    def tolist(self):
        return [self.left, self.top, self.right, self.bottom]

    @staticmethod
    def to_center_base(bboxes):
        '''  x1,y1,x2,y2 format to x_center,y_center width, height'''
        return torch.stack([
            (bboxes[..., 0] + bboxes[..., 2]) / 2,
            (bboxes[..., 1] + bboxes[..., 3]) / 2,
            bboxes[..., 2] - bboxes[..., 0],
            bboxes[..., 3] - bboxes[..., 1]], dim=-1)
    
    @staticmethod
    def from_center_base(center_based_bboxes):
        '''x_center,y_center width, height to x1,y1,x2,y2 format'''
        return torch.stack([
                center_based_bboxes[...,0] - center_based_bboxes[...,2]/2,
                center_based_bboxes[...,1] - center_based_bboxes[...,3]/2,
                center_based_bboxes[...,0] + center_based_bboxes[...,2]/2,
                center_based_bboxes[...,1] + center_based_bboxes[...,3]/2,
                ],dim=-1)
    
    @staticmethod
    def calc_transformer(ori_bboxes, target_bboxes):
        '''
        Reparameterization 
        Centers:
            tx=(x−xa)/wa and ty=(y−ya)/ha
        Height and width offsets:
            tw=log(w/wa) and th=log(h/ha)
        
        x_a = center_based_ori_bboxes[..., 0]
        y_a = center_based_ori_bboxes[..., 1]
        w_a = center_based_ori_bboxes[..., 2]
        h_a = center_based_ori_bboxes[..., 3]
        
        x = center_based_target_bboxes[..., 0]
        y = center_based_target_bboxes[..., 1]
        w = center_based_target_bboxes[..., 2]
        h = center_based_target_bboxes[..., 3]
        '''
        center_based_ori_bboxes = BBox.to_center_base(ori_bboxes)
        center_based_target_bboxes = BBox.to_center_base(target_bboxes)
        transformers = torch.stack([
            (center_based_target_bboxes[..., 0] - center_based_ori_bboxes[..., 0]) / center_based_ori_bboxes[..., 2],
            (center_based_target_bboxes[..., 1] - center_based_ori_bboxes[..., 1]) / center_based_ori_bboxes[..., 3],
            torch.log(center_based_target_bboxes[..., 2] / center_based_ori_bboxes[..., 2]),
            torch.log(center_based_target_bboxes[..., 3] / center_based_ori_bboxes[..., 3])
        ], dim=-1)
    
        return transformers

    @staticmethod
    def apply_transformer(ori_bboxes, transformers):
        '''
        Applying the transformation to the ori_bboxes
        ori_bboxes in x1,y1,x2,y2 format
        '''

        center_based_ori_bboxes = BBox.to_center_base(ori_bboxes)
        center_based_target_bboxes = torch.stack([
            transformers[..., 0] * center_based_ori_bboxes[..., 2] + center_based_ori_bboxes[..., 0],
            transformers[..., 1] * center_based_ori_bboxes[..., 3] + center_based_ori_bboxes[..., 1],
            torch.exp(transformers[..., 2]) * center_based_ori_bboxes[..., 2],
            torch.exp(transformers[..., 3]) * center_based_ori_bboxes[..., 3]
        ], dim=-1)
        '''
        Changing it (x1,y1,x2,y2) format
        '''
        target_bboxes = BBox.from_center_base(center_based_target_bboxes)
        return target_bboxes
    
    @staticmethod
    def iou(source, other):
        '''
        Calculation of the IOU (Intersection over Union)
        source (x1,y1,x2,y2) shape(bs,anchors,4)
        other (x1,y1,x2,y2) shape(bs,anchors,4)
        '''
        source, other = source.unsqueeze(dim=-2).repeat(1, 1, other.shape[-2], 1), \
                        other.unsqueeze(dim=-3).repeat(1, source.shape[-2], 1, 1)
        source_area = (source[..., 2] - source[..., 0]) * (source[..., 3] - source[..., 1])
        other_area = (other[..., 2] - other[..., 0]) * (other[..., 3] - other[..., 1])

        intersection_left = torch.max(source[..., 0], other[..., 0])
        intersection_top = torch.max(source[..., 1], other[..., 1])
        intersection_right = torch.min(source[..., 2], other[..., 2])
        intersection_bottom = torch.min(source[..., 3], other[..., 3])
        intersection_width = torch.clamp(intersection_right - intersection_left, min=0)
        intersection_height = torch.clamp(intersection_bottom - intersection_top, min=0)
        intersection_area = intersection_width * intersection_height
        return intersection_area / (source_area + other_area - intersection_area)
        
    @staticmethod
    def inside(bboxes, left, top, right, bottom):
        '''
        Return 1 if bboxes lies inside of image, 0 otherwise
        '''
        return ((bboxes[..., 0] >= left) * (bboxes[..., 1] >= top) *
                (bboxes[..., 2] <= right) * (bboxes[..., 3] <= bottom))
        
    @staticmethod
    def clip(bboxes, left, top, right, bottom):
        bboxes[..., [0, 2]] = bboxes[..., [0, 2]].clamp(min=left, max=right)
        bboxes[..., [1, 3]] = bboxes[..., [1, 3]].clamp(min=top, max=bottom)
        return bboxes
    

            
        
        
    
