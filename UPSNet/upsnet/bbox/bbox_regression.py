# ---------------------------------------------------------------------------
# Unified Panoptic Segmentation Network
#
# Modifications Copyright (c) 2019 Uber Technologies, Inc.
# --------------------------------------------------------
# Based on:
# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2016 by Contributors
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Modified by Yuwen Xiong, from py-faster-rcnn (https://github.com/rbgirshick/py-faster-rcnn)
# --------------------------------------------------------


"""
This file has functions about generating bounding box regression targets
"""

import numpy as np
import math
import cv2

from upsnet.config.config import config
from .bbox_transform import bbox_overlaps, bbox_transform


def compute_bbox_regression_targets(rois, overlaps, labels):
    """
    given rois, overlaps, gt labels, compute bounding box regression targets
    :param rois: roidb[i]['boxes'] k * 4
    :param overlaps: roidb[i]['max_overlaps'] k * 1
    :param labels: roidb[i]['max_classes'] k * 1
    :return: targets[i][class, dx, dy, dw, dh] k * 5
    """
    # Ensure ROIs are floats
    rois = rois.astype(np.float, copy=False)

    # Sanity check
    if len(rois) != len(overlaps):
        print('bbox regression: this should not happen')

    # Indices of ground-truth ROIs
    gt_inds = np.where(overlaps == 1)[0]
    if len(gt_inds) == 0:
        print('something wrong : zero ground truth rois')
    # Indices of examples for which we try to make predictions
    ex_inds = np.where(overlaps >= config.train.bbox_regression_thresh)[0]

    # Get IoU overlap between each ex ROI and gt ROI
    ex_gt_overlaps = bbox_overlaps(rois[ex_inds, :], rois[gt_inds, :])

    # Find which gt ROI each ex ROI has max overlap with:
    # this will be the ex ROI's gt target
    gt_assignment = ex_gt_overlaps.argmax(axis=1)
    gt_rois = rois[gt_inds[gt_assignment], :]
    ex_rois = rois[ex_inds, :]

    targets = np.zeros((rois.shape[0], 5), dtype=np.float32)
    targets[ex_inds, 0] = labels[ex_inds]
    targets[ex_inds, 1:] = bbox_transform(ex_rois, gt_rois)
    return targets

def compute_mask_and_label(ex_rois, ex_labels, ins_seg, class_id):
    # assert os.path.exists(seg_gt), 'Path does not exist: {}'.format(seg_gt)
    # im = Image.open(seg_gt)
    # pixel = list(im.getdata())
    # pixel = np.array(pixel).reshape([im.size[1], im.size[0]])
    rois = ex_rois
    n_rois = ex_rois.shape[0]
    label = ex_labels
    mask_target = np.zeros((n_rois, 28, 28), dtype=np.int8)
    mask_label = np.zeros((n_rois), dtype=np.int8)
    for n in range(n_rois):
        target = ins_seg[int(rois[n, 1]): int(rois[n, 3]), int(rois[n, 0]): int(rois[n, 2])]
        ids = np.unique(target)
        ins_id = 0
        max_count = 0
        for id in ids:
            if math.floor(id / 1000) == class_id[int(label[int(n)])]:
                px = np.where(ins_seg == int(id))
                x_min = np.min(px[1])
                y_min = np.min(px[0])
                x_max = np.max(px[1])
                y_max = np.max(px[0])
                x1 = max(rois[n, 0], x_min)
                y1 = max(rois[n, 1], y_min)
                x2 = min(rois[n, 2], x_max)
                y2 = min(rois[n, 3], y_max)
                iou = (x2 - x1) * (y2 - y1)
                iou = iou / ((rois[n, 2] - rois[n, 0]) * (rois[n, 3] - rois[n, 1])
                             + (x_max - x_min) * (y_max - y_min) - iou)
                if iou > max_count:
                    ins_id = id
                    max_count = iou

        if max_count == 0:
            continue
        # print max_count
        mask = np.zeros(target.shape)
        idx = np.where(target == ins_id)
        mask[idx] = 1
        mask = cv2.resize(mask, (28, 28), interpolation=cv2.INTER_NEAREST)

        mask_target[n] = mask
        mask_label[n] = label[int(n)]
    return mask_target, mask_label

def compute_bbox_mask_targets_and_label(rois, overlaps, labels, seg, class_id):
    """
    given rois, overlaps, gt labels, seg, compute bounding box mask targets
    :param rois: roidb[i]['boxes'] k * 4
    :param overlaps: roidb[i]['max_overlaps'] k * 1
    :param labels: roidb[i]['max_classes'] k * 1
    :return: targets[i][class, dx, dy, dw, dh] k * 5
    """
    # Ensure ROIs are floats
    rois = rois.astype(np.float, copy=False)

    # Sanity check
    if len(rois) != len(overlaps):
        print('bbox regression: this should not happen')

    # Indices of ground-truth ROIs
    gt_inds = np.where(overlaps == 1)[0]
    if len(gt_inds) == 0:
        print('something wrong : zero ground truth rois')
    # Indices of examples for which we try to make predictions
    ex_inds = np.where(overlaps >= config.train.bbox_regression_thresh)[0]

    # Get IoU overlap between each ex ROI and gt ROI
    ex_gt_overlaps = bbox_overlaps(rois[ex_inds, :], rois[gt_inds, :])


    # Find which gt ROI each ex ROI has max overlap with:
    # this will be the ex ROI's gt target
    gt_assignment = ex_gt_overlaps.argmax(axis=1)
    gt_rois = rois[gt_inds[gt_assignment], :]
    ex_rois = rois[ex_inds, :]

    mask_targets, mask_label = compute_mask_and_label(ex_rois, labels[ex_inds], seg, class_id)
    return mask_targets, mask_label, ex_inds


def add_bbox_regression_targets(roidb):
    """
    given roidb, add ['bbox_targets'] and normalize bounding box regression targets
    :param roidb: roidb to be processed. must have gone through imdb.prepare_roidb
    :return: means, std variances of targets
    """
    print('add bounding box regression targets')
    assert len(roidb) > 0
    assert 'max_classes' in roidb[0]

    num_images = len(roidb)
    num_classes = 2 if config.network.class_agnostic else roidb[0]['gt_overlaps'].shape[1]

    for im_i in range(num_images):
        rois = roidb[im_i]['boxes']
        max_overlaps = roidb[im_i]['max_overlaps']
        max_classes = roidb[im_i]['max_classes']
        roidb[im_i]['bbox_targets'] = compute_bbox_regression_targets(rois, max_overlaps, max_classes, config)

    if config.train.bbox_normalization_precomputed:
        # use fixed / precomputed means and stds instead of empirical values
        means = np.tile(np.array(config.train.bbox_means), (num_classes, 1))
        stds = np.tile(np.array(config.train.bbox_stds), (num_classes, 1))
    else:
        # compute mean, std values
        class_counts = np.zeros((num_classes, 1)) + 1e-14
        sums = np.zeros((num_classes, 4))
        squared_sums = np.zeros((num_classes, 4))
        for im_i in range(num_images):
            targets = roidb[im_i]['bbox_targets']
            for cls in range(1, num_classes):
                cls_indexes = np.where(targets[:, 0] > 0)[0] if config.network.class_agnostic else np.where(targets[:, 0] == cls)[0]
                if cls_indexes.size > 0:
                    class_counts[cls] += cls_indexes.size
                    sums[cls, :] += targets[cls_indexes, 1:].sum(axis=0)
                    squared_sums[cls, :] += (targets[cls_indexes, 1:] ** 2).sum(axis=0)

        means = sums / class_counts
        # var(x) = E(x^2) - E(x)^2
        stds = np.sqrt(squared_sums / class_counts - means ** 2)

    print('bbox target means:')
    print(means)
    print(means[1:, :].mean(axis=0))  # ignore bg class
    print('bbox target stdevs:')
    print(stds)
    print(stds[1:, :].mean(axis=0))  # ignore bg class


    # normalized targets
    for im_i in range(num_images):
        targets = roidb[im_i]['bbox_targets']
        for cls in range(1, num_classes):
            cls_indexes = np.where(targets[:, 0] > 0) if config.network.class_agnostic else np.where(targets[:, 0] == cls)[0]
            roidb[im_i]['bbox_targets'][cls_indexes, 1:] -= means[cls, :]
            roidb[im_i]['bbox_targets'][cls_indexes, 1:] /= stds[cls, :]

    return means.ravel(), stds.ravel()


def expand_bbox_regression_targets(bbox_targets_data, num_classes):
    """
    expand from 5 to 4 * num_classes; only the right class has non-zero bbox regression targets
    :param bbox_targets_data: [k * 5]
    :param num_classes: number of classes
    :return: bbox target processed [k * 4 num_classes]
    bbox_weights ! only foreground boxes have bbox regression computation!
    """
    classes = bbox_targets_data[:, 0]
    if config.network.class_agnostic:
        num_classes = 2
    bbox_targets = np.zeros((classes.size, 4 * num_classes), dtype=np.float32)
    bbox_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    indexes = np.where(classes > 0)[0]
    for index in indexes:
        cls = classes[index]
        start = int(4 * 1 if cls > 0 else 0) if config.network.class_agnostic else int(4 * cls)
        end = start + 4
        bbox_targets[index, start:end] = bbox_targets_data[index, 1:]
        bbox_weights[index, start:end] = config.train.bbox_weights
    return bbox_targets, bbox_weights

