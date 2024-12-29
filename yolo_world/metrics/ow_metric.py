import os
import sys
import tempfile
import copy
import logging
from typing import Optional
import numpy as np
import xml.etree.ElementTree as ET
from collections import OrderedDict, defaultdict
from functools import lru_cache

import numpy as np

from mmyolo.registry import METRICS
from mmdet.utils import ConfigType
from mmengine.logging import MMLogger
from mmengine.evaluator import BaseMetric

from ..datasets.owodb_const import *


np.set_printoptions(threshold=sys.maxsize)

@METRICS.register_module()
class OpenWorldMetric(BaseMetric):
    """
    Evaluate Pascal VOC style AP for Pascal VOC dataset.
    It contains a synchronization, therefore has to be called from all ranks.

    Note that the concept of AP can be implemented in different ways and may not
    produce identical results. This class mimics the implementation of the official
    Pascal VOC Matlab API, and should produce similar but not identical results to the
    official API.
    """
    default_prefix: Optional[str] = 'owod'

    def __init__(self,
                 data_root: str,
                 dataset_name: str,
                 owod_cfg: ConfigType = None,
                 threshold: float = 0.0,
                 save_rets: bool = False,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None):
        """
        Args:
            dataset_name (str): name of the dataset, e.g., "MOWODB"
        """
        super().__init__(collect_device=collect_device, prefix=prefix)

        self._anno_file_template = os.path.join(data_root, "Annotations", dataset_name, "{}.xml")
        self._image_set_path = os.path.join(data_root, "ImageSets", dataset_name, owod_cfg.split + ".txt")

        self._is_2007 = False
        self.threshold = threshold
        self.save_rets = save_rets
        self._logger = MMLogger.get_current_instance()
        
        self.prev_intro_cls = owod_cfg.PREV_INTRODUCED_CLS
        self.cur_intro_cls = owod_cfg.CUR_INTRODUCED_CLS
        self.total_num_class = owod_cfg.num_classes
        self.unknown_class_index = self.total_num_class - 1
        self.num_seen_classes = self.prev_intro_cls + self.cur_intro_cls
        self.known_classes = list(VOC_COCO_CLASS_NAMES[dataset_name][:self.num_seen_classes])
        self._class_names = self.known_classes + UNK_CLASS

    def process(self, data_batch: dict, data_samples):
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of data samples that
                contain annotations and predictions.
        """
        for data_sample in data_samples:
            pred = data_sample['pred_instances']
            pred_bboxes = pred['bboxes'].cpu().numpy()
            pred_scores = pred['scores'].cpu().numpy()
            pred_labels = pred['labels'].cpu().numpy()

            det = []
            for box, score, label in zip(pred_bboxes, pred_scores, pred_labels):
                if label == -100:
                    continue
                if score > self.threshold or label == self.unknown_class_index:
                    xmin, ymin, xmax, ymax = box
                    xmin += 1
                    ymin += 1
                    det.append([label, data_sample['img_id'], score, xmin, ymin, xmax, ymax])
            self.results.append(det)

    def compute_avg_precision_at_many_recall_level_for_unk(self, precisions, recalls):
        precs = {}
        for r in range(1, 10):
            r = r/10
            p = self.compute_avg_precision_at_a_recall_level_for_unk(precisions, recalls, recall_level=r)
            precs[r] = p
        return precs

    def compute_avg_precision_at_a_recall_level_for_unk(self, precisions, recalls, recall_level=0.5):
        precs = {}
        for iou, recall in recalls.items():
            prec = []
            for cls_id, rec in enumerate(recall):
                if cls_id == self.unknown_class_index and len(rec)>0:
                    p = precisions[iou][cls_id][min(range(len(rec)), key=lambda i: abs(rec[i] - recall_level))]
                    prec.append(p)
            if len(prec) > 0:
                precs[iou] = np.mean(prec)
            else:
                precs[iou] = 0
        return precs

    def compute_WI_at_many_recall_level(self, recalls, tp_plus_fp_cs, fp_os):
        wi_at_recall = {}
        for r in range(1, 10):
            r = r/10
            wi = self.compute_WI_at_a_recall_level(recalls, tp_plus_fp_cs, fp_os, recall_level=r)
            wi_at_recall[r] = wi
        return wi_at_recall

    def compute_WI_at_a_recall_level(self, recalls, tp_plus_fp_cs, fp_os, recall_level=0.5):
        wi_at_iou = {}
        for iou, recall in recalls.items():
            tp_plus_fps = []
            fps = []
            for cls_id, rec in enumerate(recall):
                if cls_id in range(self.num_seen_classes) and len(rec) > 0:
                    index = min(range(len(rec)), key=lambda i: abs(rec[i] - recall_level))
                    tp_plus_fp = tp_plus_fp_cs[iou][cls_id][index]
                    tp_plus_fps.append(tp_plus_fp)
                    fp = fp_os[iou][cls_id][index]
                    fps.append(fp)
            if len(tp_plus_fps) > 0:
                wi_at_iou[iou] = np.mean(fps) / np.mean(tp_plus_fps)
            else:
                wi_at_iou[iou] = 0
        return wi_at_iou

    def compute_metrics(self, results: list):
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        
        # get predictions by class
        predictions = defaultdict(list)
        for dets in results:
            for det in dets:
                cls, image_id, score, xmin, ymin, xmax, ymax  = det
                xmin += 1
                ymin += 1
                predictions[cls].append(
                    f"{image_id} {score:.3f} {xmin:.1f} {ymin:.1f} {xmax:.1f} {ymax:.1f}"
                )

        with tempfile.TemporaryDirectory(prefix="pascal_voc_eval_") as dirname:
            res_file_template = os.path.join(dirname, "{}.txt")

            aps = defaultdict(list)  # iou -> ap per class
            recs = defaultdict(list)
            precs = defaultdict(list)
            all_recs = defaultdict(list)
            all_precs = defaultdict(list)
            unk_det_as_knowns = defaultdict(list)
            num_unks = defaultdict(list)
            tp_plus_fp_cs = defaultdict(list)
            fp_os = defaultdict(list)

            num_kn_pred = 0
            num_unk_pred = 0
            for cls_id, cls_name in enumerate(self._class_names):
                # write class predictions to template file
                lines = predictions.get(cls_id, [])
                if cls_name in self.known_classes:
                    num_kn_pred = num_kn_pred + len(lines)
                else:
                    num_unk_pred = num_unk_pred + len(lines)

                with open(res_file_template.format(cls_name), "w") as f:
                    f.write("\n".join(lines))

                thresh = 50
                rec, prec, ap, unk_det_as_known, num_unk, tp_plus_fp_closed_set, fp_open_set = voc_eval(
                    res_file_template,
                    self._anno_file_template,
                    self._image_set_path,
                    cls_name,
                    ovthresh=thresh / 100.0,
                    use_07_metric=self._is_2007,
                    known_classes=self.known_classes
                )

                aps[thresh].append(ap * 100)
                unk_det_as_knowns[thresh].append(unk_det_as_known)
                num_unks[thresh].append(num_unk)
                all_precs[thresh].append(prec)
                all_recs[thresh].append(rec)
                tp_plus_fp_cs[thresh].append(tp_plus_fp_closed_set)
                fp_os[thresh].append(fp_open_set)
                try:
                    recs[thresh].append(rec[-1] * 100)
                    precs[thresh].append(prec[-1] * 100)
                except:
                    recs[thresh].append(0)
                    precs[thresh].append(0)

                if cls_id < self.num_seen_classes:
                    print(f"{cls_name:40s}[{cls_id:02d}]: AP{thresh}={aps[thresh][-1]:.3f}, #pred={len(lines)}")
                else:
                    print(f"{cls_name:40s}[{cls_id:02d}]: AR{thresh}={recs[thresh][-1]:.3f}, #pred={len(lines)}")

            self._logger.info("known classes has " + str(num_kn_pred) + " predictions.")
            self._logger.info("unknown classes has " + str(num_unk_pred) + " predictions.")

        wi = self.compute_WI_at_many_recall_level(all_recs, tp_plus_fp_cs, fp_os)
        self._logger.info('Wilderness Impact: ' + str(wi[0.8]))

        # avg_precision_unk = self.compute_avg_precision_at_many_recall_level_for_unk(all_precs, all_recs)
        # self._logger.info('avg_precision: ' + str(avg_precision_unk))

        ret = OrderedDict()
        # mAP = {iou: np.mean(x) for iou, x in aps.items()}
        # ret["bbox"] = {"AP": np.mean(list(mAP.values())), "AP50": mAP[50]}

        total_num_unk_det_as_known = {iou: np.sum(x) for iou, x in unk_det_as_knowns.items()}
        total_num_unk = num_unks[50][0]
        self._logger.info('Absolute OSE (total_num_unk_det_as_known): ' + str(total_num_unk_det_as_known))
        self._logger.info('total_num_unk ' + str(total_num_unk))

        # Extra logging of class-wise APs
        # self._logger.info(self._class_names)
        # self._logger.info("AP50: " + str(['%.1f' % x for x in aps[50]]))
        # self._logger.info("Precisions50: " + str(['%.1f' % x for x in precs[50]]))
        # self._logger.info("Recall50: " + str(['%.1f' % x for x in recs[50]]))

        if self.prev_intro_cls > 0:
            self._logger.info("Prev class AP50: " + str(np.mean(aps[50][:self.prev_intro_cls])))
            self._logger.info("Prev class Precisions50: " + str(np.mean(precs[50][:self.prev_intro_cls])))
            self._logger.info("Prev class Recall50: " + str(np.mean(recs[50][:self.prev_intro_cls])))

        self._logger.info("Current class AP50: " + str(np.mean(aps[50][self.prev_intro_cls:self.num_seen_classes])))
        self._logger.info("Current class Precisions50: " + str(np.mean(precs[50][self.prev_intro_cls:self.num_seen_classes])))
        self._logger.info("Current class Recall50: " + str(np.mean(recs[50][self.prev_intro_cls:self.num_seen_classes])))
        self._logger.info("Known AP50: " + str(np.mean(aps[50][:self.num_seen_classes])))
        self._logger.info("Known Precisions50: " + str(np.mean(precs[50][:self.num_seen_classes])))
        self._logger.info("Known Recall50: " + str(np.mean(recs[50][:self.num_seen_classes])))
        self._logger.info("Unknown AP50: " + str(aps[50][-1]))
        self._logger.info("Unknown Precisions50: " + str(precs[50][-1]))
        self._logger.info("Unknown Recall50: " + str(recs[50][-1]))

        ret = {
            "U-Recall": recs[50][-1],
            "WI": round(wi[0.8][50], 5),
            "A-OSE": int(total_num_unk_det_as_known[50]),
            "PK": np.mean(aps[50][:self.prev_intro_cls]) if self.prev_intro_cls > 0 else 0.0,
            "CK": np.mean(aps[50][self.prev_intro_cls:self.num_seen_classes]),
            "Both": np.mean(aps[50][:self.num_seen_classes]),
        }
        
        if self.save_rets:
            with open('eval_outputs.txt', 'a') as f:
                f.write(f"{ret['WI']:8.5f} {ret['A-OSE']:8d} {ret['U-Recall']:8.2f} {ret['PK']:8.2f} {ret['CK']:8.2f} {ret['Both']:8.2f}" + '\n')

        return ret


##############################################################################
#
# Below code is modified from
# https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/voc_eval.py
# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------

"""Python implementation of the PASCAL VOC devkit's AP evaluation code."""


@lru_cache(maxsize=None)
def parse_rec(filename, known_classes):
    """Parse a PASCAL VOC xml file."""
    tree = ET.parse(filename)

    objects = []
    for obj in tree.findall("object"):
        obj_struct = {}
        cls_name = obj.find("name").text
        if cls_name in VOC_CLASS_NAMES_COCOFIED:
            cls_name = BASE_VOC_CLASS_NAMES[VOC_CLASS_NAMES_COCOFIED.index(cls_name)]
        if cls_name not in known_classes:
            cls_name = 'unknown'
        obj_struct["name"] = cls_name
        # obj_struct["pose"] = obj.find("pose").text
        # obj_struct["truncated"] = int(obj.find("truncated").text)
        obj_struct['difficult'] = int(obj.find('difficult').text) if obj.find('difficult') is not None else 0
        bbox = obj.find("bndbox")
        obj_struct["bbox"] = [
            int(bbox.find("xmin").text),
            int(bbox.find("ymin").text),
            int(bbox.find("xmax").text),
            int(bbox.find("ymax").text),
        ]
        objects.append(obj_struct)

    return objects


def voc_ap(rec, prec, use_07_metric=False):
    """Compute VOC AP given precision and recall. If use_07_metric is true, uses
    the VOC 07 11-point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath, annopath, imagesetfile, classname, ovthresh=0.5, use_07_metric=False, known_classes=None):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])

    Top level function that does the PASCAL VOC evaluation.

    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name

    # first load gt
    # read list of images
    with open(imagesetfile, "r") as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    imagenames_filtered = []
    # load annots
    recs = {}
    mapping = {}   # follow RandBox to deduplicate
    for imagename in imagenames:
        rec = parse_rec(annopath.format(imagename), tuple(known_classes))
        # if rec is not None:
        if rec is not None and int(imagename) not in mapping:
            recs[imagename] = rec
            imagenames_filtered.append(imagename)
            mapping[int(imagename)] = imagename

    imagenames = imagenames_filtered

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj["name"] == classname]
        bbox = np.array([x["bbox"] for x in R])
        difficult = np.array([x["difficult"] for x in R]).astype(np.bool_)
        # difficult = np.array([False for x in R]).astype(np.bool_)  # treat all "difficult" as GT
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {"bbox": bbox, "difficult": difficult, "det": det}
    
    # read dets
    detfile = detpath.format(classname)
    with open(detfile, "r") as f:
        lines = f.readlines()

    splitlines = [x.strip().split(" ") for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines]).reshape(-1, 4)

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    for d in range(nd):
        # R = class_recs[image_ids[d]]
        R = class_recs[mapping[int(image_ids[d])]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R["bbox"].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
            ih = np.maximum(iymax - iymin + 1.0, 0.0)
            inters = iw * ih

            # union
            uni = (
                (bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0)
                + (BBGT[:, 2] - BBGT[:, 0] + 1.0) * (BBGT[:, 3] - BBGT[:, 1] + 1.0)
                - inters
            )

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R["difficult"][jmax]:
                if not R["det"][jmax]:
                    tp[d] = 1.0
                    R["det"][jmax] = 1
                else:
                    fp[d] = 1.0
        else:
            fp[d] = 1.0

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / (float(npos) + 1e-5)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)


    '''
    Computing Absolute Open-Set Error (A-OSE) and Wilderness Impact (WI)
                                    ===========    
    Absolute OSE = # of unknown objects classified as known objects of class 'classname'
    WI = FP_openset / (TP_closed_set + FP_closed_set)
    '''

    # Finding GT of unknown objects
    unknown_class_recs = {}
    n_unk = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj["name"] == 'unknown']
        bbox = np.array([x["bbox"] for x in R])
        difficult = np.array([x["difficult"] for x in R]).astype(np.bool_)
        det = [False] * len(R)
        n_unk = n_unk + sum(~difficult)
        unknown_class_recs[imagename] = {"bbox": bbox, "difficult": difficult, "det": det}

    if classname == 'unknown':
        return rec, prec, ap, 0, n_unk, None, None

    # Go down each detection and see if it has an overlap with an unknown object.
    # If so, it is an unknown object that was classified as known.
    is_unk = np.zeros(nd)
    for d in range(nd):
        # R = unknown_class_recs[image_ids[d]]
        R = unknown_class_recs[mapping[int(image_ids[d])]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R["bbox"].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
            ih = np.maximum(iymax - iymin + 1.0, 0.0)
            inters = iw * ih

            # union
            uni = (
                (bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0)
                + (BBGT[:, 2] - BBGT[:, 0] + 1.0) * (BBGT[:, 3] - BBGT[:, 1] + 1.0)
                - inters
            )

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            is_unk[d] = 1.0

    is_unk_sum = np.sum(is_unk)

    tp_plus_fp_closed_set = tp+fp
    fp_open_set = np.cumsum(is_unk)

    return rec, prec, ap, is_unk_sum, n_unk, tp_plus_fp_closed_set, fp_open_set


def print_total_annatations(imagenames, known_classes, recs):
    known_classes_un = known_classes + ['known'] + ['unknown']
    total_ann = [0 for _ in known_classes_un]
    for imagename in imagenames:
        for obj in recs[imagename]:
            if obj["name"] in known_classes:
                total_ann[known_classes.index(obj["name"])] += 1
                total_ann[-2] += 1
            else:
                total_ann[-1] += 1
    print('valid annotations:')
    for i in range(0, len(known_classes_un), 3):
        line = ""
        for j in range(3):
            if i + j < len(known_classes_un):
                category, count = known_classes_un[i+j], total_ann[i+j]
                line += f"{category.ljust(15)} | {str(count).rjust(5)} | "
        print(line)

