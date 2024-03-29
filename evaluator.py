from typing import Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from coco import CocoDetection 
class Evaluator(object):
    def __init__(self,dataset,path_to_results_dir):
        super().__init__()
        self._dataset = dataset
        self._dataloader = DataLoader(dataset, batch_size=1,shuffle=False,num_workers=8,pin_memory=True)
        self._device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        self._path_to_results_dir = path_to_results_dir 
    def evaluate(self,model):
            with torch.no_grad():
                all_image_ids, all_detection_bboxes, all_detection_classes, all_detection_probs = [], [], [], []
                for it, (image_id_batch,image_batch,scale_batch,_,_,skip) in enumerate(self._dataloader):
                    print(it)
                    if(it > len(self._dataloader)/50):
                        break
                    if(skip == 0):
                         image_batch = image_batch.to(self._device)
                         assert image_batch.shape[0] == 1
                         detection_bboxes , detection_classes, detection_probs, detection_batch_indices = model.eval().forward(image_batch)
                         '''scaling  of detection bboxes as image has been scaled'''
                         scale_batch = scale_batch[detection_batch_indices].unsqueeze(dim=-1).expand_as(detection_bboxes).to(self._device)
                         detection_bboxes = detection_bboxes / scale_batch

                         kept_indices = (detection_probs > 0.05).nonzero().view(-1)
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

