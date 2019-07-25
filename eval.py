import os
import time
import uuid
import utils
from model import Model
from coco import CocoDetection
from backbone import vgg16
from evaluator import Evaluator
from logger import Logger

def _eval(path_to_checkpoint, dataset_name, backbone_name, path_to_results_dir, config):
    dataset = CocoDetection(root=config.dataset_name, annFile = config.annFile, min_size = config.min_size, max_size = config.max_size)
    evaluator = Evaluator(dataset,path_to_results_dir)
    Logger.i('Found {:d} samples'.format(len(dataset)))
    backbone = vgg16(pretrained=False)
    model = Model(backbone = backbone, num_classes = config.num_classes, pooler_mode = 'pooling',
    anchor_ratios = config.anchor_ratios, anchor_sizes = config.anchor_sizes, 
    rpn_pre_nms_top_n = config.RPN_PRE_NMS_TOP_N, 
    rpn_post_nms_top_n = config.RPN_POST_NMS_TOP_N,
    anchor_smooth_l1_loss_beta = config.ANCHOR_SMOOTH_L1_LOSS_BETA,
    proposal_smooth_l1_loss_beta = config.PROPOSAL_SMOOTH_L1_LOSS_BETA
    ).cuda()
    model.load(path_to_checkpoint)
    Logger.i('Start evaluation')
    mean_ap, detail = evaluator.evaluate(model)
    Logger.i('Done')
    Logger.i('mean_AP = {:.4f}'.format(mean_ap))
    Logger.i('\n' + detail)

if __name__ == '__main__':
    config = utils.Params('config_eval.json')
    path_to_checkpoint = 'outputs/model_resnet101.pth'
#    path_to_results_dir = os.path.join(os.path.dirname(path_to_checkpoint), 'results-{:s}-{:s}-{:s}'.format(
 #                   time.strftime('%Y%m%d%H%M%S'), path_to_checkpoint.split(os.path.sep)[-1].split(os.path.curdir)[0],
  #                  str(uuid.uuid4()).split('-')[0]))
    path_to_results_dir = 'eval/'
    path_to_data_dir = 'COCO'
    config_str = [(key,value) for (key,value) in config.dict.items()]
    Logger.initialize(os.path.join(path_to_results_dir,'eval.log'))
    Logger.i('Hypyerparameters and arguments')
    config_list = [(key,value) for (key,value) in config.dict.items()]
    for pair in config_list :
        Logger.i(pair)
    _eval(path_to_checkpoint, config.dataset_name, config.backbone_name, path_to_results_dir, config)


