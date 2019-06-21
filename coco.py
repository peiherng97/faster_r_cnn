# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from vision import VisionDataset
from PIL import Image
import os
import os.path
import numpy as np
from torchvision.transforms import transforms
import torch
import matplotlib.pyplot as plt

class CocoDetection(VisionDataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, root, annFile, min_size, max_size, transform=None, target_transform=None, transforms=None):
        super(CocoDetection, self).__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.min_size = min_size
        self.max_size = max_size

    @staticmethod
    def preprocess(img, min_size, max_size):
        """ 
        Takes a PIL format image and scales it accordingly to the min and max width defined below
        Both the longer and shorter side should be less than max_size and min_size
        """
        if(min_size > max_size):
            raise Exception('min_size should not exceed max_size')
            
        width, height = img.size
        minDim = min(width,height)
        maxDim = max(width,height)
        scale_shorter_side = min_size/minDim
        scale_longer_side = maxDim * scale_shorter_side
        if(scale_longer_side > max_size):
            scale = max_size/maxDim
        else:
            scale = scale_shorter_side
        transform = transforms.Compose([
                transforms.Resize((round(img.height*scale),round(img.width * scale))),
                transforms.ToTensor(),
#                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        img = transform(img)
        return scale,img
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        annotation = coco.loadAnns(ann_ids)
        
        bboxes_gt = [torch.Tensor(ann['bbox']) for ann in annotation] 
        category_id = [ann['category_id'] for ann in annotation] 
        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        scale,img = self.preprocess(img=img,min_size=600,max_size=1000)
        bboxes_gt = [torch.mul(bbox_coords,scale) for bbox_coords in bboxes_gt]
        return img_id, img, scale, bboxes_gt, category_id


    def __len__(self):
        return len(self.ids)
    

    
root = 'COCO/val2017'
annFile = 'COCO/annotations/instances_val2017.json'
coco = CocoDetection(root = root, annFile = annFile)
#temp = coco.__getitem__(10)
#npimg = temp[1].numpy()
#plt.imshow(np.transpose(npimg, (1,2,0)),interpolation='nearest')
