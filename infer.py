import argparse
import os
import json
import random
import torch
import utils
from PIL import ImageDraw
from torchvision.transforms import transforms
from bbox import BBox
from model_resnet101 import Model
from coco import CocoDetection
from backbone import Resnet101
from pycocotools.coco import COCO
import numpy as np
from pycocotools.cocoeval import COCOeval

def _infer(path_to_input_image, path_to_output_image, path_to_checkpoint, dataset_class, backbone, model,imgid):
    backbone = backbone(pretrained=False)
    model = model(backbone = backbone, num_classes = config.num_classes, pooler_mode = 'align', anchor_ratios = config.anchor_ratios, anchor_sizes = config.anchor_sizes, rpn_pre_nms_top_n = config.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n = config.RPN_POST_NMS_TOP_N)
    model.load_state_dict(torch.load(path_to_checkpoint))

    with torch.no_grad():
        image = transforms.Image.open(path_to_input_image)
        scale, image_tensor = dataset_class.preprocess(image, config.min_size, config.max_size)
        detection_bboxes, detection_classes, detection_probs, _ = model.eval().forward(image_tensor.unsqueeze(dim=0))
        detection_bboxes /= scale
        kept_indices = detection_probs > 0.05
        print(detection_probs.max())
        detection_bboxes = detection_bboxes[kept_indices]
        detection_classes = detection_classes[kept_indices]
        detection_probs = detection_probs[kept_indices]
        draw = ImageDraw.Draw(image)
        results_val = []
        for bbox, cls, prob in zip(detection_bboxes.tolist(), detection_classes.tolist(), detection_probs.tolist()):
            results_val.append(
                {
                    'image_id' : int(imgid),
                    'category_id' : cls,
                    'bbox' : [
                        bbox[0],
                        bbox[1],
                        bbox[2]-bbox[0],
                        bbox[3]-bbox[1],
                        ],
                    'score': 0.95
                    }
                )

            color = random.choice(['red', 'green', 'blue', 'yellow', 'purple', 'white'])
            bbox = BBox(left=bbox[0], top=bbox[1], right=bbox[2], bottom=bbox[3])
            category = dataset_class.LABEL_TO_CATEGORY_DICT[cls]
            draw.rectangle(((bbox.left, bbox.top), (bbox.right, bbox.bottom)), outline=color)
            draw.text((bbox.left,bbox.top),text=f'{category:s}{prob:.3f}',fill=color)
        with open('results_val.json','w') as f:
                json.dump(results_val,f)

    image.save(path_to_output_image + 'test1.png')
    image_gt = transforms.Image.open(path_to_input_image)
    draw_gt = ImageDraw.Draw(image_gt)
    ann_ids = coco.getAnnIds(imgIds=imgid,iscrowd = None)
    annotation = coco.loadAnns(ann_ids)
    bboxes_gt = [ann['bbox'] for ann in annotation]  
    category_id = [ann['category_id'] for ann in annotation] 
    results = []
    for bbox, cls in zip(bboxes_gt,category_id):
            results.append(
                {
                    'image_id' : int(imgid),
                    'category_id' : cls,
                    'bbox' : [
                        bbox[0],
                        bbox[1],
                        bbox[2],
                        bbox[3],
                        ],
                    'score': 0.95
                    }
                )

            color = random.choice(['red', 'green', 'blue', 'yellow', 'purple', 'white'])
            bbox = BBox(left=bbox[0], top=bbox[1], right=bbox[2]+bbox[0], bottom=bbox[3]+bbox[1])
            category = dataset_class.LABEL_TO_CATEGORY_DICT[cls]
            draw_gt.rectangle(((bbox.left, bbox.top), (bbox.right, bbox.bottom)), outline=color)
            draw_gt.text((bbox.left,bbox.top),text=f'{category:s}',fill=color)
    image_gt.save(path_to_output_image + 'gt.png')
    with open('results.json','w') as f:
                json.dump(results,f)
    annType = 'bbox'
    path_to_coco_dir = 'COCO'
    path_to_annotations_dir = os.path.join(path_to_coco_dir, 'annotations')
    path_to_annotation = os.path.join(path_to_annotations_dir, 'instances_val2017.json')
    cocoGt = COCO(path_to_annotation)
    cocoDt = cocoGt.loadRes('results_val.json')
    cocoEval = COCOeval(cocoGt,cocoDt,annType)
    cocoEval.params.imgIds  = int(imgid)
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()



    print(f'Output image is saved to {path_to_output_image}')

if __name__ == '__main__':
    config = utils.Params('config_resnet101_eval.json')
    annFile = config.annFile
    coco = COCO(annFile)
    catids = coco.getCatIds('people')
    img_ids = coco.getImgIds(catIds=catids)
    #imgid = 139
    #print(np.random.randint(len(img_ids),size=1))
    imgid = img_ids[np.random.randint(len(img_ids),size=1)[0]]
    print(imgid)
    path_to_input_image = coco.loadImgs(imgid)[0]['file_name']
    path_to_input_image = 'COCO/images/val2017/' + path_to_input_image
    path_to_output_image = 'inference/output'
    path_to_checkpoint = 'outputs/model_resnet101.pth'
    if os.path.exists(path_to_output_image):
        print('inference path detected')
    else:
        os.mkdir(path_to_output_image)
    _infer(path_to_input_image, path_to_output_image, path_to_checkpoint, CocoDetection, Resnet101, Model,imgid) 
