import os
import time
import uuid
import utils
import numpy as np
from model_resnet101 import Model
from coco_animal import CocoDetection
from backbone import Resnet101
from logger import Logger
import torch
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler


class Evaluator(object):
    def __init__(self,dataset,path_to_results_dir):
        super().__init__()
        self._dataset = dataset
        train_indices = np.arange(100)
        self._dataloader = DataLoader(dataset, batch_size=1,shuffle=False,num_workers=8,pin_memory=True)
        self._device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        self._path_to_results_dir = path_to_results_dir 
    def evaluate(self,model):
            with torch.no_grad():
                all_image_ids, all_detection_bboxes, all_detection_classes, all_detection_probs = [], [], [], []
                for it, (image_id_batch, image_batch, scale_batch, bboxes_batch, labels_batch) in enumerate(self._dataloader):

                         image_batch = image_batch.to(self._device)
                         assert image_batch.shape[0] == 1
                         detection_bboxes , detection_classes, detection_probs, detection_batch_indices = model.eval().forward(image_batch)
                         '''scaling  of detection bboxes as image has been scaled'''
                         scale_batch = scale_batch[detection_batch_indices].unsqueeze(dim=-1).expand_as(detection_bboxes).to(self._device)
                         detection_bboxes = detection_bboxes / scale_batch
                         kept_indices = (detection_probs > 0.7).nonzero().view(-1)
                         detection_bboxes = detection_bboxes[kept_indices]
                         detection_classes = detection_classes[kept_indices]
                         detection_probs = detection_probs[kept_indices]
                         detecton_batch_indices = detection_batch_indices[kept_indices]
                         all_detection_bboxes.extend(detection_bboxes.tolist())
                         all_detection_classes.extend(detection_classes.tolist())
                         all_detection_probs.extend(detection_probs.tolist())
                         all_image_ids.extend([image_id_batch[i] for i in detection_batch_indices])
            mean_ap, detail = self._dataset.evaluate(self._path_to_results_dir, all_image_ids, all_detection_bboxes, all_detection_classes, all_detection_probs)
            return mean_ap, detail


if __name__ == '__main__':
    config = utils.Params('config_resnet101_eval.json')
    path_to_checkpoint = 'outputs/model_resnet101.pth'
    path_to_data_dir = 'datadir'
    path_to_results_dir = './'
    dataset = CocoDetection(root=config.dataset_name, annFile = config.annFile, min_size = config.min_size, max_size = config.max_size)
    evaluator = Evaluator(dataset,path_to_results_dir)
    backbone = Resnet101(pretrained=False)
    model = Model(backbone = backbone, num_classes = config.num_classes, pooler_mode = 'align',
    anchor_ratios = config.anchor_ratios, anchor_sizes = config.anchor_sizes, 
    rpn_pre_nms_top_n = config.RPN_PRE_NMS_TOP_N, 
    rpn_post_nms_top_n = config.RPN_POST_NMS_TOP_N,
    anchor_smooth_l1_loss_beta = config.ANCHOR_SMOOTH_L1_LOSS_BETA,
    proposal_smooth_l1_loss_beta = config.PROPOSAL_SMOOTH_L1_LOSS_BETA
    ).cuda()
    model.load_state_dict(torch.load(path_to_checkpoint))
    model.eval()
    mean_ap, detail = evaluator.evaluate(model)
    print(mean_ap)
    print(detail)



