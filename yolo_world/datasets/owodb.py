# partly taken from  https://github.com/pytorch/vision/blob/master/torchvision/datasets/voc.py
import functools

import os
import copy
from mmdet.utils import ConfigType
from mmdet.datasets import BaseDetDataset
from mmyolo.datasets.yolov5_coco import BatchShapePolicyDataset
from mmyolo.registry import DATASETS

import xml.etree.ElementTree as ET
from mmengine.logging import MMLogger

from .owodb_const import *

@DATASETS.register_module()
class OWODDataset(BatchShapePolicyDataset, BaseDetDataset):
    """`OWOD in Pascal VOC format <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Detection Dataset.

    Args:
        root (string): Root directory of the VOC Dataset.
        year (string, optional): The dataset year, supports years 2007 to 2012.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
            (default: alphabetic indexing of VOC's 20 classes).
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, required): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """
    METAINFO = {
        'classes': (),
        'palette': None,
    }

    def __init__(self,
                 data_root: str,
                 dataset: str = 'MOWODB',
                 image_set: str='train',
                 owod_cfg: ConfigType = None,
                 training_strategy: int = 0,
                 **kwargs):

        self.images = []
        self.annotations = []
        self.imgids = []
        self.imgid2annotations = {}
        self.image_set_fns = []

        self.image_set = image_set
        self.dataset=dataset
        self.CLASS_NAMES = VOC_COCO_CLASS_NAMES[dataset]
        self.task_num = owod_cfg.task_num
        self.owod_cfg = owod_cfg
        
        self._logger = MMLogger.get_current_instance()

        # training strategy
        self.training_strategy = training_strategy
        if "test" not in image_set:
            if training_strategy == 0:
                self._logger.info(f"Training strategy: OWOD")
            elif training_strategy == 1:
                self._logger.info(f"Training strategy: ORACLE")
            else:
                raise ValueError(f"Invalid training strategy: {training_strategy}")

        OWODDataset.METAINFO['classes'] = self.CLASS_NAMES
        
        self.data_root=str(data_root)
        annotation_dir = os.path.join(self.data_root, 'Annotations', dataset)
        image_dir = os.path.join(self.data_root, 'JPEGImages', dataset)

        file_names = self.extract_fns()
        self.image_set_fns.extend(file_names)
        self.images.extend([os.path.join(image_dir, x + ".jpg") for x in file_names])
        self.annotations.extend([os.path.join(annotation_dir, x + ".xml") for x in file_names])
        self.imgids.extend(x for x in file_names)            
        self.imgid2annotations.update(dict(zip(self.imgids, self.annotations)))

        assert (len(self.images) == len(self.annotations) == len(self.imgids))

        super().__init__(**kwargs)

    def extract_fns(self):
        splits_dir = os.path.join(self.data_root, 'ImageSets')
        splits_dir = os.path.join(splits_dir, self.dataset)
        image_sets = []
        file_names = []

        if 'test' in self.image_set: # for test
            image_sets.append(self.image_set)
        else: # owod or oracle
            image_sets.append(f"t{self.task_num}_{self.image_set}")

        self.image_set_list = image_sets
        for image_set in image_sets:
            split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')
            with open(os.path.join(split_f), "r") as f:
                file_names.extend([x.strip() for x in f.readlines()])
        return file_names

    ### OWOD
    def remove_prev_class_and_unk_instances(self, target):
        # For training data. Removing earlier seen class objects and the unknown objects..
        prev_intro_cls = self.owod_cfg.PREV_INTRODUCED_CLS
        curr_intro_cls = self.owod_cfg.CUR_INTRODUCED_CLS
        valid_classes = range(prev_intro_cls, prev_intro_cls + curr_intro_cls)
        entry = copy.copy(target)
        for annotation in copy.copy(entry):
            if annotation["bbox_label"] not in valid_classes:
                entry.remove(annotation)
        return entry

    def remove_unknown_instances(self, target):
        # For finetune data. Removing the unknown objects...
        prev_intro_cls = self.owod_cfg.PREV_INTRODUCED_CLS
        curr_intro_cls = self.owod_cfg.CUR_INTRODUCED_CLS
        valid_classes = range(0, prev_intro_cls+curr_intro_cls)
        entry = copy.copy(target)
        for annotation in copy.copy(entry):
            if annotation["bbox_label"] not in valid_classes:
                entry.remove(annotation)
        return entry

    def label_known_class_and_unknown(self, target):
        # For test and validation data.
        # Label known instances the corresponding label and unknown instances as unknown.
        prev_intro_cls = self.owod_cfg.PREV_INTRODUCED_CLS
        curr_intro_cls = self.owod_cfg.CUR_INTRODUCED_CLS
        total_num_class = self.owod_cfg.num_classes
        known_classes = range(0, prev_intro_cls+curr_intro_cls)
        entry = copy.copy(target)
        for annotation in copy.copy(entry):
        # for annotation in entry:
            if annotation["bbox_label"] not in known_classes:
                annotation["bbox_label"] = total_num_class - 1
        return entry

    def load_data_list(self):
        data_list = []
        self._logger.info(f"Loading {self.dataset} from {self.image_set_list}...")
        for i, img_id in enumerate(self.imgids):
            raw_data_info = dict(
                img_path=self.images[i],
                img_id=img_id,
            )
            parsed_data_info = self.parse_data_info(raw_data_info)
            data_list.append(parsed_data_info)

        self._logger.info(f"{self.dataset} Loaded, {len(data_list)} images in total")
        return data_list
    
    def parse_data_info(self, raw_data_info):
        data_info = copy.copy(raw_data_info)
        img_id = data_info["img_id"]
        tree = ET.parse(self.imgid2annotations[img_id])

        instances = []
        for obj in tree.findall("object"):
            cls = obj.find("name").text

            if cls in VOC_CLASS_NAMES_COCOFIED:
                cls = BASE_VOC_CLASS_NAMES[VOC_CLASS_NAMES_COCOFIED.index(cls)]
            try:
                bbox_label = self.CLASS_NAMES.index(cls)
            except ValueError:
                continue # ignore 'ego' class in nu-OWODB
            bbox = obj.find("bndbox")
            bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
            bbox[0] -= 1.0
            bbox[1] -= 1.0
            instance = dict(
                bbox_label=bbox_label,
                bbox=bbox,
                area=(bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
                ignore_flag=0,
            )
            instances.append(instance)

        if 'train' in self.image_set:
            if self.training_strategy == 1: # oracle
                instances = self.label_known_class_and_unknown(instances)
            else: # owod
                instances = self.remove_prev_class_and_unk_instances(instances)
        elif 'test' in self.image_set:
            instances = self.label_known_class_and_unknown(instances)
        elif 'ft' in self.image_set:
            instances = self.remove_unknown_instances(instances)
            
        data_info.update(
            height=int(tree.findall("./size/height")[0].text),
            width=int(tree.findall("./size/width")[0].text),
            instances=instances,
        )

        return data_info

    def filter_data(self):
        """Filter annotations according to filter_cfg.

        Returns:
            List[dict]: Filtered results.
        """
        if self.test_mode:
            return self.data_list

        filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False) \
            if self.filter_cfg is not None else False
        min_size = self.filter_cfg.get('min_size', 0) \
            if self.filter_cfg is not None else 0

        valid_data_infos = []
        for i, data_info in enumerate(self.data_list):
            width = data_info['width']
            height = data_info['height']
            if filter_empty_gt and len(data_info['instances']) == 0:
                continue
            if min(width, height) >= min_size:
                valid_data_infos.append(data_info)

        return valid_data_infos