# *-* coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np

try:
  import cPickle as pickle
except ImportError:
  import pickle
import os
import math
import sys
import h5py

from lib.utils.timer import Timer
from lib.model.nms_wrapper import nms_test as nms
from lib.utils.blob import im_list_to_blob

from lib.model.bbox_transform_cpu import clip_boxes, bbox_transform_inv
from lib.model.config import cfg, cfg_from_file, cfg_from_list, get_output_model_dir, get_output_dir

from lib.nets.vgg16 import VGG16
from lib.nets.network import FasterRCNN
from lib.nets.resnet import Resnet
import xml.etree.ElementTree as ET
import torch

def parse_rec(filename):
  """ Parse a PASCAL VOC xml file """
  tree = ET.parse(filename)
  objects = []
  for obj in tree.findall('object'):
    obj_struct = {}
    obj_struct['name'] = obj.find('name').text
    obj_struct['pose'] = obj.find('pose').text
    obj_struct['truncated'] = int(obj.find('truncated').text)
    obj_struct['difficult'] = int(obj.find('difficult').text)
    bbox = obj.find('bndbox')
    obj_struct['bbox'] = [int(bbox.find('xmin').text),
                          int(bbox.find('ymin').text),
                          int(bbox.find('xmax').text),
                          int(bbox.find('ymax').text)]
    objects.append(obj_struct)

  return objects

def _get_image_blob(im):
  """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
  im_orig = im.astype(np.float32, copy=True)
  im_orig -= cfg.PIXEL_MEANS

  im_shape = im_orig.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])

  processed_ims = []
  im_scale_factors = []

  for target_size in cfg.TEST.SCALES:
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
      im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, np.array(im_scale_factors)


def _get_blobs(im):
  """Convert an image and RoIs within that image into network inputs."""
  blobs = {}
  blobs['data'], im_scale_factors = _get_image_blob(im)

  return blobs, im_scale_factors


def _clip_boxes(boxes, im_shape):
  """Clip boxes to image boundaries."""
  # x1 >= 0
  boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
  # y1 >= 0
  boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
  # x2 < im_shape[1]
  boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
  # y2 < im_shape[0]
  boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
  return boxes


def _rescale_boxes(boxes, inds, scales):
  """Rescale boxes according to image rescaling."""
  for i in range(boxes.shape[0]):
    boxes[i, :] = boxes[i, :] / scales[int(inds[i])]

  return boxes


def im_detect(net, im):
  blobs, im_scales = _get_blobs(im)
  assert len(im_scales) == 1, "Only single-image batch implemented"

  im_blob = blobs['data']
  blobs['im_info'] = np.array([im_blob.shape[1], im_blob.shape[2], im_scales[0]], dtype=np.float32)

  scores, bbox_pred, rois = net(blobs['data'], blobs['im_info'])
  scores = scores.data.cpu().numpy()
  bbox_pred = bbox_pred.data.cpu().numpy()
  rois = rois.data.cpu().numpy()

  boxes = rois[:, 1:5] / im_scales[0]
  scores = np.reshape(scores, [scores.shape[0], -1])
  bbox_pred = np.reshape(bbox_pred, [bbox_pred.shape[0], -1])
  if cfg.TEST.BBOX_REG:
    # Apply bounding-box regression deltas
    box_deltas = bbox_pred
    pred_boxes = bbox_transform_inv(boxes, box_deltas)
    pred_boxes = _clip_boxes(pred_boxes, im.shape)
  else:
    # Simply repeat the boxes, once for each class
    pred_boxes = np.tile(boxes, (1, scores.shape[1]))

  return scores, pred_boxes


class SolverWrapper(object):
  def __init__(self, network, model_dir, output_dir, classes):
    self.model_dir = model_dir
    self.net = network
    self.output_dir = output_dir
    self.classes = classes
    self.num_class = len(classes)

  def load_check_point(self, step):
    net = self.net
    filename = os.path.join(self.model_dir, 'fasterRcnn_iter_{}.h5'.format(step))
    print('Restoring model snapshots from {:s}'.format(filename))

    if not os.path.exists(filename):
      print('The checkPoint is not Right')
      sys.exit(1)

    # load model
    h5f = h5py.File(filename, mode='r')
    for k, v in net.state_dict().items():
      param = torch.from_numpy(np.asarray(h5f[k]))
      v.copy_(param)


  def prepare_construct(self, resume_iter):
    # init network
    self.net.init_fasterRCNN()

    # load model
    if resume_iter:
      self.load_check_point(resume_iter)

    # model
    self.net.eval()
    if cfg.CUDA_IF:
      self.net.cuda()

  def test_model(self, img_path, max_per_image=100, thresh=0.):

    im = cv2.imread(img_path)

    all_boxes = [[] for _ in range(self.num_class)]

    scores, boxes = im_detect(self.net, im)

    # skip j = 0, because it's the background class
    for j in range(1, self.num_class):
      inds = np.where(scores[:, j] > thresh)[0]
      or_cls_scores = scores[inds, :]
      cls_scores = scores[inds, j]
      cls_boxes = boxes[inds, j * 4:(j + 1) * 4]
      cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
        .astype(np.float32, copy=False)
      keep = nms(cls_dets, cfg.TEST.NMS) if cls_dets.size > 0 else []
      cls_dets = cls_dets[keep, :]
      cls_scores = or_cls_scores[keep, :]
      all_boxes[j] = [cls_scores, cls_dets]

    result = None
    # Limit to max_per_image detections *over all classes*
    if max_per_image > 0:
      image_scores = np.hstack([all_boxes[j][1][:, -1] for j in range(1, self.num_class)])

      # Log num of proposal > 1
      if len(image_scores) > 1:
        cls_dets = [all_boxes[j] for j in xrange(1, self.num_class) if len(all_boxes[j][1]) != 0]

      if len(image_scores) > max_per_image:
        image_thresh = np.sort(image_scores)[-max_per_image]
        for j in range(1, self.num_class):
          keep = np.where(all_boxes[j][1][:, -1] >= image_thresh)[0]
          all_boxes[j] = [all_boxes[j][0][keep, :], all_boxes[j][1][keep, :]]
        cls_dets = [all_boxes[j] for j in xrange(1, self.num_class) if len(all_boxes[j][1]) != 0]
        if (len(cls_dets) != 0):
          result = cls_dets
        else:
          # No result
          pass
      else:
        cls_dets = [all_boxes[j] for j in xrange(1, self.num_class) if len(all_boxes[j][1]) != 0]
        if (len(cls_dets) != 0):
          result = cls_dets
        else:
          # No result
          pass

    print('Finish')

    if result is not None:
      scores, bbox = result[0]
      scores = scores[:, 1:]
      pred_label = np.argmax(scores, axis=1)[0] + 1
      pred_label_score = np.max(scores, axis=1)[0]
      print('=='*10)
      print('Predict: {} ({}%)'.format(self.classes[pred_label], round(pred_label_score, 3)*100))
      # top1
      pred_bbox = bbox[0].astype(np.int32)
      print('Cervix Area: \n--top left point ({}, {})\n--bottom right point ({}, {})'.
            format(pred_bbox[0], pred_bbox[1], pred_bbox[2], pred_bbox[3]))

      # pred_label: 类别序号
      # pred_label_score: 置信度
      # pred_bbox: 宫颈面位置左上角坐标+右下角坐标
      return pred_label, pred_label_score, pred_bbox

    else:
      print('Can not judge!')
      return None, None, None



if __name__ == '__main__':
  os.environ['CUDA_VISIBLE_DEVICES'] = '2'

  net = 'res101'
  model_dir = 'cervix_train/res101'
  resume = 30000
  cfg_file = './experiments/cfgs/cervix_res101.yml'
  output_dir = './demo_output'

  img_path = './data/cervix/Images/23000016_2.jpg'
  xml_path = './data/cervix/Xmls/23000016_2.xml'



  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  # load cfg
  if cfg_file is not None:
    cfg_from_file(cfg_file)

  model_dir = os.path.join(cfg.ROOT_DIR, 'model', cfg.EXP_DIR, model_dir, cfg.TRAIN.SNAPSHOT_PREFIX)

  # load network
  if net == 'vgg16':
    net = FasterRCNN(VGG16(feat_strdie=(16,),
                           anchor_scales=cfg.ANCHOR_SCALES,
                           anchor_ratios=cfg.ANCHOR_RATIOS), cfg.CLASSES)
    cfg.TRAIN.INIT_WAY = 'vgg'
  elif net == 'res18':
    net = FasterRCNN(Resnet(resnet_type=18, feat_strdie=(16,),
                            anchor_scales=cfg.ANCHOR_SCALES,
                            anchor_ratios=cfg.ANCHOR_RATIOS), cfg.CLASSES)
    cfg.TRAIN.INIT_WAY = 'resnet'
  elif net == 'res50':
    net = FasterRCNN(Resnet(resnet_type=50, feat_strdie=(16,),
                            anchor_scales=cfg.ANCHOR_SCALES,
                            anchor_ratios=cfg.ANCHOR_RATIOS), cfg.CLASSES)
    cfg.TRAIN.INIT_WAY = 'resnet'
  elif net == 'res101':
    net = FasterRCNN(Resnet(resnet_type=101, feat_strdie=(16,),
                            anchor_scales=cfg.ANCHOR_SCALES,
                            anchor_ratios=cfg.ANCHOR_RATIOS), cfg.CLASSES)
    cfg.TRAIN.INIT_WAY = 'resnet'
  else:
    raise NotImplementedError

  solver = SolverWrapper(network=net, model_dir=model_dir, output_dir=output_dir, classes=cfg.CLASSES)
  solver.prepare_construct(resume)
  solver.test_model(img_path, max_per_image=1, thresh=0.05)

  def get_gt(xml_path):
    object_s = parse_rec(xml_path)
    gt_label = [obj['name'].lower().strip() for obj in object_s]
    label = gt_label[0]
    object_bb = [obj['bbox'] for obj in object_s]
    return label, object_bb

  print(get_gt(xml_path))

