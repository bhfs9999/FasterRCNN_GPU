# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Zheqi he, Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.model.eval import eval_net
from lib.model.config import cfg, cfg_from_file, cfg_from_list, get_output_model_dir, get_output_dir
from lib.datasets.factory import get_cervex_db
import argparse
import pprint
import time, os, sys

from lib.nets.vgg16 import VGG16
from lib.nets.network import FasterRCNN
from lib.nets.resnet import Resnet


import torch

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
  parser.add_argument('--cfg', dest='cfg_file',
            help='optional config file', default='./experiments/cfgs/vgg16.yml', type=str)
  parser.add_argument('--model', dest='model_check_point',
            help='model to test',
            default=None, type=str)
  parser.add_argument('--model_path', dest='model_path',
                      help='model path',
                      default='voc_2007_trainval', type=str)
  parser.add_argument('--imdb', dest='imdb_name',
            help='dataset to test',
            default='voc_2007_test', type=str)
  parser.add_argument('--image_type', dest='image_type_name',
                      help='image type',
                      default='acid', type=str)
  parser.add_argument('--data_type', dest='data_type_name',
                      help='data type',
                      default='cervix', type=str)
  parser.add_argument('--comp', dest='comp_mode', help='competition mode',
            action='store_true')
  parser.add_argument('--num_dets', dest='max_per_image',
            help='max number of detections per image',
            default=1, type=int)
  parser.add_argument('--tag', dest='tag',
                        help='tag of the model',
                        default='', type=str)
  parser.add_argument('--net', dest='net',
                      help='vgg16, res50, res101, res152, mobile',
                      default='res50', type=str)
  parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)

  args = parser.parse_args()
  return args

if __name__ == '__main__':
  args = parse_args()

  # args.model_check_point = 70000
  # args.max_per_image = 300
  # args.net ='vgg16'
  # args.tag = 'test'
  # args.model_path = '/home/yxd/projects/cervix/FasterRCNN_torch/model/vgg16/voc_2007_trainval/vgg16/vgg16_faster_rcnn'
  # args.imdb_name = 'voc_2007_test'
  # os.environ['CUDA_VISIBLE_DEVICES'] = '4'

  print('Called with args:')
  print(args)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)

  image_type = None
  if args.image_type_name == 'salt':
    image_type = '_1'
  elif args.image_type_name == 'acid':
    image_type = '_2'
  elif args.image_type_name == 'iodine':
    image_type = '_3'
  else:
    print('image_type must in salt, acid, iodine')
    raise NotImplementedError

  data_type = None
  if args.data_type_name == 'cervix':
    data_type = 'cervix'
  elif args.data_type_name == 'cervixMore':
    data_type = 'cervixMore'
  else:
    print('data_type is not defined')
    raise NotImplementedError

  # if has model, get the name from it
  if args.model_check_point:
    resume_iter = args.model_check_point
  else:
    print('can not load model')
    sys.exit(1)


  imdb = get_cervex_db(args.imdb_name, image_type, data_type)
  imdb.competition_mode(args.comp_mode)

  # model directory where the summaries are saved during training
  model_dir = args.model_path
  model_dir = os.path.join(cfg.ROOT_DIR, 'model', cfg.EXP_DIR, args.model_path, cfg.TRAIN.SNAPSHOT_PREFIX)
  print('Model will load from `{:s}`'.format(model_dir))

  # output directory where the models are saved
  output_dir = get_output_dir(imdb, args.tag)
  output_dir = os.path.join(output_dir, cfg.TRAIN.SNAPSHOT_PREFIX)
  print('Output will be saved to `{:s}`'.format(output_dir))

  # load network
  if args.net == 'vgg16':
    print(imdb.classes)
    net = FasterRCNN(VGG16(feat_strdie=(16,),
                           anchor_scales=cfg.ANCHOR_SCALES,
                           anchor_ratios=cfg.ANCHOR_RATIOS), imdb.classes)
    cfg.TRAIN.INIT_WAY = 'vgg'
  elif args.net == 'res18':
    net = FasterRCNN(Resnet(resnet_type=18, feat_strdie=(16,),
                           anchor_scales=cfg.ANCHOR_SCALES,
                           anchor_ratios=cfg.ANCHOR_RATIOS), imdb.classes)
    cfg.TRAIN.INIT_WAY = 'resnet'
  elif args.net == 'res50':
    net = FasterRCNN(Resnet(resnet_type=50, feat_strdie=(16,),
                           anchor_scales=cfg.ANCHOR_SCALES,
                           anchor_ratios=cfg.ANCHOR_RATIOS), imdb.classes)
    cfg.TRAIN.INIT_WAY = 'resnet'
  elif args.net == 'res101':
    net = FasterRCNN(Resnet(resnet_type=101, feat_strdie=(16,),
                           anchor_scales=cfg.ANCHOR_SCALES,
                           anchor_ratios=cfg.ANCHOR_RATIOS), imdb.classes)
    cfg.TRAIN.INIT_WAY = 'resnet'
  else:
    raise NotImplementedError

  eval_net(net, imdb, resume_iter, model_dir, output_dir, max_per_image=args.max_per_image, thresh=0.05)
