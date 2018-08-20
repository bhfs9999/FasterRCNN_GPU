
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

from lib.model.config import cfg, get_output_dir
from lib.model.bbox_transform_cpu import clip_boxes, bbox_transform_inv

import torch
import xml.etree.ElementTree as ET


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
  # if cfg.TEST.BBOX_REG:
  # Apply bounding-box regression deltas
  box_deltas = bbox_pred
  pred_boxes = bbox_transform_inv(boxes, box_deltas)
  pred_boxes = _clip_boxes(pred_boxes, im.shape)
  # else:
  # Simply repeat the boxes, once for each class
  # pred_boxes = np.tile(boxes, (1, scores.shape[1]))


  return scores, pred_boxes, boxes


# def apply_nms(all_boxes, thresh):
#   """Apply non-maximum suppression to all predicted boxes output by the
#   test_net method.
#   """
#   num_classes = len(all_boxes)
#   num_images = len(all_boxes[0])
#   nms_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
#   for cls_ind in range(num_classes):
#     for im_ind in range(num_images):
#       dets = all_boxes[cls_ind][im_ind]
#       if dets == []:
#         continue
#
#       x1 = dets[:, 0]
#       y1 = dets[:, 1]
#       x2 = dets[:, 2]
#       y2 = dets[:, 3]
#       scores = dets[:, 4]
#       inds = np.where((x2 > x1) & (y2 > y1))[0]
#       dets = dets[inds, :]
#       if dets == []:
#         continue
#
#       keep = nms(torch.from_numpy(dets), thresh).numpy()
#       if len(keep) == 0:
#         continue
#       nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
#   return nms_boxes

class SolverWrapper(object):
  def __init__(self, network, imdb, model_dir, output_dir):
    self.model_dir = model_dir
    self.net = network
    self.imdb = imdb
    self.output_dir = output_dir

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

    # Set the random seed
    torch.manual_seed(cfg.RNG_SEED)
    np.random.seed(cfg.RNG_SEED)

    # load model
    if resume_iter:
      self.load_check_point(resume_iter)

    # model
    self.net.eval()
    if cfg.CUDA_IF:
      self.net.cuda()

  def eval_model(self, resume_iter, max_per_image=100, thresh=0.):
    """Test a Fast R-CNN network on an image database."""
    num_images = len(self.imdb.image_index)
    # all detections are collected into:
    #  all_boxes[cls][image] = N x 5 array of detections in
    #  (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(self.imdb.num_classes)]

    output_dir = os.path.join(self.output_dir, 'fasterRcnn_iter_{}'.format(resume_iter))
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}

    image_example_list = ['_'.join(img.split('_')[:-1]) for img in self.imdb.image_index]
    pred_objects = dict()

    # record how many image can not be bounded by a bbox
    images_can_bound_count = 0
    images_all_count = 0
    ignore_pred_list = list()
    confuse_pred_dict = dict()

    for i in range(num_images):
      imgs_path = self.imdb.image_path_at(i)
      imgs = cv2.imread(imgs_path)

      _t['im_detect'].tic()
      scores, boxes, rois_boxes = im_detect(self.net, imgs)

      _t['im_detect'].toc()

      _t['misc'].tic()

      # skip j = 0, because it's the background class
      for j in range(1, self.imdb.num_classes):
        inds = np.where(scores[:, j] > thresh)[0]
        or_cls_scores = scores[inds, :]
        cls_scores = scores[inds, j]
        cls_boxes = boxes[inds, j * 4:(j + 1) * 4]
        cls_rois_boxes = rois_boxes[inds, :]
        cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
          .astype(np.float32, copy=False)
        keep = nms(cls_dets, cfg.TEST.NMS) if cls_dets.size > 0 else []
        cls_dets = cls_dets[keep, :]
        cls_scores = or_cls_scores[keep, :]
        cls_rois_boxes = cls_rois_boxes[keep, :]
        all_boxes[j][i] = [cls_scores, cls_dets, cls_rois_boxes]

      images_all_count += 1

      # Limit to max_per_image detections *over all classes*
      if max_per_image > 0:
        image_scores = np.hstack([all_boxes[j][i][1][:, -1] for j in range(1, self.imdb.num_classes)])

        # Log num of proposal > 1
        if len(image_scores) > 1:
          cls_dets = [all_boxes[j][i] for j in xrange(1, self.imdb.num_classes) if len(all_boxes[j][i][1]) != 0]
          confuse_pred_dict[image_example_list[i]] = cls_dets

        if len(image_scores) > max_per_image:
          image_thresh = np.sort(image_scores)[-max_per_image]
          for j in range(1, self.imdb.num_classes):
            keep = np.where(all_boxes[j][i][1][:, -1] >= image_thresh)[0]
            all_boxes[j][i] = [all_boxes[j][i][0][keep, :], all_boxes[j][i][1][keep, :],
                               all_boxes[j][i][2][keep, :]]
          cls_dets = [all_boxes[j][i] for j in xrange(1, self.imdb.num_classes) if len(all_boxes[j][i][1]) != 0]
          if (len(cls_dets) != 0):
            images_can_bound_count += 1
            pred_objects[image_example_list[i]] = cls_dets
          else:
            ignore_pred_list.append(image_example_list[i])
        else:
          cls_dets = [all_boxes[j][i] for j in xrange(1, self.imdb.num_classes) if len(all_boxes[j][i][1]) != 0]
          if (len(cls_dets) != 0):
            images_can_bound_count += 1
            pred_objects[image_example_list[i]] = cls_dets
          else:
            ignore_pred_list.append(image_example_list[i])
      _t['misc'].toc()

      if self.model_dir is not None:
        print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
              .format(i + 1, num_images, _t['im_detect'].toc(average=False),
                      _t['misc'].toc(average=False)))

    # write prediction result
    with open(os.path.join(output_dir, 'predict_objects.pkl'), 'wb') as f:
      pickle.dump(pred_objects, f)
    with open(os.path.join(output_dir, 'ignore_objects.pkl'), 'wb') as f:
      pickle.dump(ignore_pred_list, f)
    with open(os.path.join(output_dir, 'confuse_objects.pkl'), 'wb') as f:
      pickle.dump(confuse_pred_dict, f)
    if self.model_dir is not None:
      print('Evaluating Classification')
      print('Save in: ' + output_dir)
      print('Have predict Images {} / {} !!'.format(images_can_bound_count, images_all_count))
      print('Confused predicted images {} / {} !!'.format(len(confuse_pred_dict), images_all_count))
      print('Ignored predicted images {} / {} !!'.format(len(ignore_pred_list), images_all_count))
    annopath = os.path.join(
      self.imdb._data_path,
      'Xmls',
      '{:s}.xml')
    image_example_set = pred_objects.keys()

    cachedir = os.path.join(cfg.DATA_DIR, 'annotations_cache', self.imdb._data_type, self.imdb._image_set+self.imdb._image_type)

    gt_objects = self.load_GT_labels(annopath, [img for img in self.imdb.image_index], cachedir)
    metrics_cls = self.evaluate_classifications(pred_objects, gt_objects, image_example_set, self.imdb._image_type, self.imdb, output_dir)
    metrics_reg = self.evaluate_regressions(pred_objects, gt_objects, image_example_set, self.imdb, output_dir, 0.5)
    return metrics_cls, metrics_reg

  @staticmethod
  def load_GT_labels(annopath, imagesetfile, cachedir):
    # load GT labels
    if not os.path.isdir(cachedir):
      os.makedirs(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')

    imagenames = imagesetfile

    if not os.path.isfile(cachefile):
      # load annots
      recs = {}
      for i, imagename in enumerate(imagenames):
        recs[imagename] = parse_rec(annopath.format(imagename))
        if i % 100 == 0:
          print('Reading annotation for {:d}/{:d}'.format(
            i + 1, len(imagenames)))
      # save
      print('Saving cached annotations to {:s}'.format(cachefile))
      with open(cachefile, 'w') as f:
        pickle.dump(recs, f)
    else:
      # load
      with open(cachefile, 'r') as f:
        recs = pickle.load(f)
    return recs

  def evaluate_regressions(self, pred_objects, gt_objects, imageset, imdb, output_dir, ovthresh=0.5):
    tp_result = []
    for img_name in imageset:
      scores, bbox, bbox_rois = pred_objects[img_name][0]
      object_s_1 = gt_objects[img_name + imdb._image_type]
      object_bb_1 = [obj['bbox'] for obj in object_s_1]

      # top1
      bbox_1 = bbox[0]
      bbox_2 = bbox_rois[0]
      overlaps_1 = self.overlaps_cal(object_bb_1, bbox_1)
      overlaps_2 = self.overlaps_cal(object_bb_1, bbox_2)
      if_tp_1, _ = self.tp_judge(overlaps_1, ovthresh)
      if_tp_2, _ = self.tp_judge(overlaps_2, ovthresh)
      tp_result.append([if_tp_1, if_tp_2, img_name])
      # tp_result.append([if_tp_1, img_name])
    precision_top_1 = np.sum([line[0] for line in tp_result])
    precision_top_2 = np.sum([line[1] for line in tp_result])
    precision_down = len(tp_result)
    precion_1 = precision_top_1 * 1.0 / precision_down
    precion_2 = precision_top_2 * 1.0 / precision_down

    metric_s = dict()
    metric_s['RPN_Refine_precision'] = [precion_1, precision_top_1, precision_down]
    metric_s['RPN_precision'] = [precion_2, precision_top_2, precision_down]

    if self.model_dir is not None:
      print('=====' * 10)

      print('RPN_Refine proposal precision is : {:.4f} ({}/{})'.format(precion_1, precision_top_1, precision_down))
      print('RPN proposal precision is : {:.4f} ({}/{})'.format(precion_2, precision_top_2, precision_down))

    with open(os.path.join(output_dir, 'RPN_metric.txt'), 'wb') as f:
      for key, value in metric_s.items():
        f.write('{}: {} ({}/{})\n'.format(key, value[0], value[1], value[2]))
    with open(os.path.join(output_dir, 'RPN_result.txt'), 'wb') as f:
      for v1, v2, v3 in tp_result:
        f.write('{}:{},{}\n'.format(v3, v1, v2))
    return metric_s

  @staticmethod
  def overlaps_cal(gt_bbox, pre_bbox):
    # make sure the type is np.array
    '''
    :param gt_bbox:   shape = [n, 4]
    :param pre_bbox:  shape = [4]
    :return
        overlaps shape=[n]
    '''
    if not isinstance(gt_bbox, np.ndarray):
      BBGT = np.array(gt_bbox)
    else:
      BBGT = gt_bbox
    if not isinstance(pre_bbox, np.ndarray):
      bb = np.array(pre_bbox)
    else:
      bb = pre_bbox

    if BBGT.size > 0:
      # compute overlaps
      # intersection
      ixmin = np.maximum(BBGT[:, 0], bb[0])
      iymin = np.maximum(BBGT[:, 1], bb[1])
      ixmax = np.minimum(BBGT[:, 2], bb[2])
      iymax = np.minimum(BBGT[:, 3], bb[3])
      iw = np.maximum(ixmax - ixmin + 1., 0.)
      ih = np.maximum(iymax - iymin + 1., 0.)
      inters = iw * ih

      # union
      uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
             (BBGT[:, 2] - BBGT[:, 0] + 1.) *
             (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

      overlaps = inters / uni
      return overlaps
    else:
      return [0.0 for i in range(BBGT.shape[0])]

  @staticmethod
  def tp_judge(overlaps, ovthresh):
    ovmax = np.max(overlaps)
    jmax = np.argmax(overlaps)
    if ovmax > ovthresh:
      return True, jmax
    else:
      return False, -1

  def evaluate_classifications(self, pred_objects, gt_objects, imageset, img_index, imdb, output_dir):
    pred_labels = []
    gt_labels = []
    imageset_ar = np.array(imageset)
    for img_name in imageset:
      # more than one proposal
      if len(pred_objects[img_name]) > 1:
        print('Have More than one proposal :', len(pred_objects[img_name]))
      scores, _, _ = pred_objects[img_name][0]
      scores = scores[:, 1:]
      pred_label = np.argmax(scores, axis=1)[0] + 1
      object_s = gt_objects[img_name + img_index]
      gt_label = [imdb._class_to_ind[obj['name'].lower().strip()] for obj in object_s]
      label = gt_label[0]
      pred_labels.append(pred_label)
      gt_labels.append(label)
    pred_labels = np.array(pred_labels)
    gt_labels = np.array(gt_labels)
    right_predicts = pred_labels == gt_labels
    wrong_predicts = pred_labels != gt_labels

    # accuracy   tp+tn/(tp+fn+tn+fp)
    acc_top = np.sum(right_predicts)
    acc_down = len(gt_labels)
    Acc = acc_top / float(acc_down)
    if self.model_dir is not None:
      print('=====' * 10)
      print('Accuracy is : {:.4f}  ({}/{})'.format(Acc, acc_top, acc_down))
    metric_s = dict()
    metric_s['acc'] = [Acc, acc_top, acc_down]
    for label_name in imdb.classes:
      if label_name == '__background__':
        continue
      label_all_in_gt = gt_labels == imdb._class_to_ind[label_name]
      # recall tp/(tp+fn)
      sum_in_gt = np.sum((label_all_in_gt))
      recall_top = np.sum(right_predicts & label_all_in_gt)
      recall_down = sum_in_gt
      recall = (recall_top * 1.0 / recall_down) if sum_in_gt != 0 else 0.0

      acc_indexs = np.where((right_predicts & label_all_in_gt))[0]
      acc_images = imageset_ar[acc_indexs]
      wrong_indexs = np.where((wrong_predicts & label_all_in_gt))[0]
      wrong_images = imageset_ar[wrong_indexs]

      with open(os.path.join(output_dir, label_name + '_right.txt'), 'wb') as f:
        f.write('\n'.join(acc_images))
      with open(os.path.join(output_dir, label_name + '_wrong.txt'), 'wb') as f:
        f.write('\n'.join(wrong_images))

      if self.model_dir is not None:
        print('Recall of {}: {:.4f} ({}/{})'.format(label_name, recall, recall_top, recall_down))
      metric_s[label_name + '_recall'] = [recall, recall_top, recall_down]

      with open(os.path.join(output_dir, label_name + '_metric.txt'), 'wb') as f:
        for key, value in metric_s.items():
          f.write('{}: {} ({} / {})\n'.format(key, value[0], value[1], value[2]))
    return metric_s


def eval_net(net, imdb, resume_iter, model_dir, output_dir, max_per_image=100, thresh=0.):
  solver = SolverWrapper(network=net, imdb=imdb, model_dir=model_dir, output_dir=output_dir)
  solver.prepare_construct(resume_iter)
  solver.eval_model(resume_iter, max_per_image, thresh)

