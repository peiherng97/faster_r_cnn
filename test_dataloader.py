from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from typing import List, Tuple
import numpy as np
coco = COCO("COCO/annotations/instances_train2017.json")
img_ids_filtered = coco.getImgIds(catIds = coco.getCatIds('person'))
img_ids = list(sorted(img_ids_filtered))
img_id_ann_length = np.asarray([len(coco.getAnnIds(imgIds=img_ids,catIds=coco.getCatIds('person'))) for img_ids in img_ids])
img_id_ann_length[0] = 0
img_ids = np.asarray(img_ids)
img_ids = img_ids[img_id_ann_length > 0]
print(len(img_id_ann_length))
print(len(img_ids))
