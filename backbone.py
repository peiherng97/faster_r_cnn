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
        features = list(self.vgg16.features)
        classifier = list(self.vgg16.classifier)
        features = features[:-1]
        classifier = classifier[:-1]
        submodel = nn.Sequential(*list(features))
        hidden = nn.Sequential(*list(classifier))
        num_features_out = 512
        num_hidden_out = 4096
        '''Freezing the layers before conv3 following fast r cnn'''
        for layer in range(10):
            for p in submodel[layer].parameters():
                p.require_grad = False
                
        return submodel, hidden, num_features_out, num_hidden_out
    
    
class Resnet18(object):
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
        for parameters in [layer.parameters() for i, layer in enumerate(layers) if i <=4]:
            for parameter in parameters:
                parameter.requires_grad = False
        submodel = nn.Sequential(*layers)
        return submodel, hidden, num_features_out, num_hidden_out
    
class Resnet101(object):
    def __init__(self,pretrained=True):
        super().__init__()
        self._pretrained = pretrained
        self.resnet101 = torchvision.models.resnet101(pretrained=self._pretrained)

    def features(self):

        # list(resnet101.children()) consists of following modules
        #   [0] = Conv2d, [1] = BatchNorm2d, [2] = ReLU,
        #   [3] = MaxPool2d, [4] = Sequential(Bottleneck...),
        #   [5] = Sequential(Bottleneck...),
        #   [6] = Sequential(Bottleneck...),
        #   [7] = Sequential(Bottleneck...),
        #   [8] = AvgPool2d, [9] = Linear
        children = list(self.resnet101.children())
        features = children[:-3]
        num_features_out = 1024
        hidden = children[-3]
        num_featuers_out = 1024
        num_hidden_out = 2048

        for parameters in [feature.parameters() for i, feature in enumerate(features) if i<=4]:
            for parameter in parameters:
                parameter.requires_grad = False

        features = nn.Sequential(*features)

        return features, hidden, num_features_out, num_hidden_out

#model = vgg16()
#model.features()
