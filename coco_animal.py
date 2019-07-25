# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from typing import List,Tuple
from vision import VisionDataset
from PIL import Image
import os
import os.path
import numpy as np
import json
from torchvision.transforms import transforms
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import sys
from io import StringIO
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

    CATEGORY_TO_LABEL_DICT = {
'background': 0, 'person': 1, 'bicycle': 2, 'car': 3, 'motorcycle': 4,
'airplane': 5, 'bus': 6, 'train': 7, 'truck': 8, 'boat': 9,
'traffic light': 10, 'fire hydrant': 11, 'street sign': 12, 'stop sign': 13, 'parking meter': 14,
'bench': 15, 'bird': 16, 'cat': 17, 'dog': 18, 'horse': 19,
'sheep': 20, 'cow': 21, 'elephant': 22, 'bear': 23, 'zebra': 24,
'giraffe': 25, 'hat': 26, 'backpack': 27, 'umbrella': 28, 'shoe': 29,
'eye glasses': 30, 'handbag': 31, 'tie': 32, 'suitcase': 33, 'frisbee': 34,
'skis': 35, 'snowboard': 36, 'sports ball': 37, 'kite': 38, 'baseball bat': 39,
'baseball glove': 40, 'skateboard': 41, 'surfboard': 42, 'tennis racket': 43, 'bottle': 44,
'plate': 45, 'wine glass': 46, 'cup': 47, 'fork': 48, 'knife': 49,
'spoon': 50, 'bowl': 51, 'banana': 52, 'apple': 53, 'sandwich': 54,
'orange': 55, 'broccoli': 56, 'carrot': 57, 'hot dog': 58, 'pizza': 59,
'donut': 60, 'cake': 61, 'chair': 62, 'couch': 63, 'potted plant': 64,
'bed': 65, 'mirror': 66, 'dining table': 67, 'window': 68, 'desk': 69,
'toilet': 70, 'door': 71, 'tv': 72, 'laptop': 73, 'mouse': 74,
'remote': 75, 'keyboard': 76, 'cell phone': 77, 'microwave': 78, 'oven': 79,
'toaster': 80, 'sink': 81, 'refrigerator': 82, 'blender': 83, 'book': 84,
'clock': 85, 'vase': 86, 'scissors': 87, 'teddy bear': 88, 'hair drier': 89,
'toothbrush': 90, 'hair brush': 91
}
    
#    CATEGORY_TO_LABEL_DICT = {
#        'background': 0, 'person': 1
#    }

    
    def __init__(self, root, annFile, min_size, max_size, train = 1, transform=None, target_transform=None, transforms=None):
        super(CocoDetection, self).__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        self.coco = COCO(annFile)
        #self.catids = self.coco.getCatIds('person')
        self.ids = list(sorted(self.coco.imgs.keys()))
        #self.ids = sorted(self.coco.getImgIds(catIds=self.ids))
        img_id_ann_length = np.asarray([len(self.coco.getAnnIds(imgIds=img_ids)) for img_ids in self.ids])
        self.ids = np.asarray(self.ids,dtype=int)
        self.ids = self.ids[img_id_ann_length > 0].tolist()
        #self.ids = list(sorted(self.coco.imgs.keys()))
        self.min_size = min_size
        self.max_size = max_size
        self._train = train

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        #ann_ids = coco.getAnnIds(imgIds=img_id,catIds=self.catids, iscrowd = None)
        ann_ids = coco.getAnnIds(imgIds=img_id,iscrowd = None)
        annotation = coco.loadAnns(ann_ids)
        bboxes_gt = [ann['bbox'] for ann in annotation]  
        bboxes_gt = torch.tensor(bboxes_gt, dtype = torch.float) 
        '''x1,y1,width,height -> x1,y1,x2.y2'''
        bboxes_gt[...,2] = bboxes_gt[...,0] + bboxes_gt[...,2]
        bboxes_gt[...,3] = bboxes_gt[...,1] + bboxes_gt[...,3]
        category_id = [ann['category_id'] for ann in annotation] 
        category_id = torch.tensor(category_id, dtype=torch.long)
        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        scale,img = self.preprocess(img=img,min_size=self.min_size,max_size=self.max_size)
        bboxes_gt = bboxes_gt * scale
        return img_id, img, scale, bboxes_gt, category_id


    def __len__(self):
        return len(self.ids)
    
    @staticmethod
    def preprocess(img, min_size, max_size):
        """ 
        Takes a PIL format image and scales it accordingly to the min and max width defined below
        Both the longer and shorter side should be less than max_size and #min_size
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
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        img = transform(img)
        scale = torch.tensor(scale,dtype=torch.float)
        return scale,img

    @staticmethod
    def num_classes():
        return 2
    @staticmethod
    def padding_collate_fn(batch : List[Tuple[str, Tensor, Tensor, Tensor, Tensor]]):
        image_id_batch, image_batch, scale_batch, bboxes_batch, labels_batch = zip(*batch)
        max_image_width = max([it.shape[2] for it in image_batch])
        max_image_height = max([it.shape[1] for it in image_batch])
        max_bboxes_length = max([len(it) for it in bboxes_batch])
        max_labels_length = max([len(it) for it in labels_batch])
        
        padded_image_batch = []
        padded_bboxes_batch = []
        padded_labels_batch = []
        
        for image in image_batch:
            padded_image = F.pad(input=image, pad=(0, max_image_width - image.shape[2], 0, max_image_height - image.shape[1]))
            padded_image_batch.append(padded_image)
        for bboxes in bboxes_batch:
            padded_bboxes = torch.cat([bboxes, torch.zeros(max_bboxes_length - len(bboxes),4).to(bboxes)])
            padded_bboxes_batch.append(padded_bboxes)
        for labels in labels_batch:
            padded_labels = torch.cat([labels, torch.zeros(max_labels_length - len(labels)).to(labels)])
            padded_labels_batch.append(padded_labels)
        
        image_id_batch = list(image_id_batch)
        padded_image_batch = torch.stack(padded_image_batch, dim = 0)
        scale_batch = torch.stack(scale_batch, dim = 0)
        padded_bboxes_batch = torch.stack(padded_bboxes_batch, dim = 0)
        padded_labels_batch = torch.stack(padded_labels_batch, dim = 0)
       
        return image_id_batch, padded_image_batch, scale_batch, padded_bboxes_batch, padded_labels_batch

    def evaluate(self, path_to_results_dir: str, image_ids: List[str], bboxes: List[List[float]], classes: List[int], probs: List[float]) -> Tuple[float, str]:
        print('writing results')
        self._write_results(path_to_results_dir, image_ids, bboxes, classes, probs)
        annType = 'bbox'
       # path_to_coco_dir = os.path.join(self._path_to_data_dir, 'COCO')
        path_to_coco_dir = 'COCO'
        path_to_annotations_dir = os.path.join(path_to_coco_dir, 'annotations')
        path_to_annotation = os.path.join(path_to_annotations_dir, 'instances_val2017.json')
        cocoGt = COCO(path_to_annotation)
        cocoDt = cocoGt.loadRes('results.json')
        cocoEval = COCOeval(cocoGt,cocoDt,annType)
        cocoEval.params.imgIds  = image_ids
        cocoEval.evaluate()
        cocoEval.accumulate()
        original_stdout = sys.stdout
        string_stdout = StringIO()
        sys.stdout = string_stdout
        cocoEval.summarize()
        sys.stdout = original_stdout
        mean_ap = cocoEval.stats[0].item()
        detail = string_stdout.getvalue()
        return mean_ap, detail

    def _write_results(self,path_to_results_dir, image_ids, bboxes, classes, probs):
        results = []
        for image_id, bbox, cls, prob in zip(image_ids, bboxes, classes, probs):
            results.append(
                {
                    'image_id' : int(image_id),
                    'category_id' : cls,
                    'bbox' : [
                        bbox[0],
                        bbox[1],
                        bbox[2] - bbox[0],
                        bbox[3] - bbox[1],
                        ],
                    'score': prob
                    }
                )
        with open(os.path.join(path_to_results_dir, 'results.json'),'w') as f:
                json.dump(results,f)




#ro = 'COCO/val2017'
#annFile = 'COCO/annotations/instances_val2017.json'
#coco = CocoDetection(root = root, annFile = annFile)
#temp = coco.__getitem__(10)
#npimg = temp[1].numpy()
#plt.imshow(np.transpose(npimg, (1,2,0)),interpolation='nearest')
