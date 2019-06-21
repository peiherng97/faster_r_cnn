#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 12:13:47 2019

@author: ph97
"""
import torchvision
from torch import nn

class vgg16(object):
    
    def __init__(self,pretrained=True):
        super().__init__()
        self._pretrained = pretrained
        self.vgg16 = torchvision.models.vgg16(pretrained=self._pretrained)

    def features(self):
        '''remove last layer of maxpool, subsample of 16'''
        features = list(self.vgg16.features[:-1])
        submodel = nn.Sequential(*list(features))
        num_features_out = 512
        print(features)
        '''Freezing the layers before conv3 following fast r cnn'''
        for layer in range(10):
            for p in submodel[layer].parameters():
                p.require_grad = False
                
        return submodel, num_features_out
    
    
class ResNet18(object):
    def __init__(self,pretrained=True):
        super().__init__()
        self._pretrained = pretrained
        self.resnet18 = torchvision.models.resnet18(pretrained=self._pretrained)

    def features(self):
        children = list(self.resnet18.children())
        layers = children[:-3]
        num_features_out = 256
        hidden = children[-3]
        num_hidden_out = 512
        print(hidden)
        for parameters in [layer.parameters() for i, layer in enumerate(layers) if i <=4]:
            for parameter in parameters:
                parameter.requires_grad = False
        submodel = nn.Sequential(*layers)
        return submodel, hidden, num_features_out, num_hidden_out
    
#model = vgg16()
#model.features()
model1 = ResNet18()
model1.features()