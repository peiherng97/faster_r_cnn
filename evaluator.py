from typing import Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from coco import CocoDetection 
class Evaluator(object):
    def __init__(self,dataset,path_to_data_dir,path_to_results_dir):
        super().__init__()
        self._dataset = datset
        self._dataloader = Dataloader(dataset, batch_size=1,shuffle=False,num_workers=8,pin_memory=True)
        self._path_to_data_dir = path_to_data_dir
        self._path_to_results_dir = path_to_results_dir

    def evaluate(self,model):
            with torch.no_grad():
                for _, (image_id_batch,image_batch,scale_batch,_,_) in enumerate(self_.dataloader):
                    image_batch = image_batch.cuda()
                    assert image_batch.shape[0] == 1

                    detection_bboxes , detection_classes, detection_probs, detection_batch_indices = model.eval().forward(image_batch)
                    scale_batch = scale_batch[detection_batch_indices].unsqueeze(dim=-1).expand_as(detection_bboxes).to(device)
                    detection_bboxes = detection_bboxes / scale_batch

                    kept_indices = (detection_probs > 0.05).nonzero().view(-1)
                    detection_bboxes = detection_bboxes[kept_indices]
                    detection_classes = detection_classes[kept_classes]
                   

