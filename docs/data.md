## Preparing Data

### Dataset Directory

We put all data into the `data` directory, such as:

```bash
├── coco
│   ├── annotations
│   ├── lvis
│   ├── train2017
│   ├── val2017
├── flickr
│   ├── annotations
│   └── full_images
├── mixed_grounding
│   ├── annotations
│   ├── images
├── objects365v1
│   ├── annotations
│   ├── train
│   ├── val
├── OWOD
│   ├── JPEGImages
│   ├── Annotations
│   └── ImageSets
```
**NOTE**: We strongly suggest that you check the directories or paths in the dataset part of the config file, especially for the values `ann_file`, `data_root`, and `data_prefix`.

### Open Vocabulary Dataset

For pre-training YOLO-UniOW, we adopt several datasets as listed in the below table:

| Data | Samples | Type | Boxes  |
| :-- | :-----: | :---:| :---: | 
| Objects365v1 | 609k | detection | 9,621k |
| GQA | 621k | grounding | 3,681k |
| Flickr | 149k | grounding | 641k |

We provide the annotations of the pre-training data in the below table:

| Data | images | Annotation File |
| :--- | :------| :-------------- |
| Objects365v1 | [`Objects365 train`](https://opendatalab.com/OpenDataLab/Objects365_v1) | [`objects365_train.json`](https://opendatalab.com/OpenDataLab/Objects365_v1) |
| MixedGrounding | [`GQA`](https://nlp.stanford.edu/data/gqa/images.zip) | [`final_mixed_train_no_coco.json`](https://huggingface.co/GLIPModel/GLIP/tree/main/mdetr_annotations/final_mixed_train_no_coco.json) |
| Flickr30k | [`Flickr30k`](https://shannon.cs.illinois.edu/DenotationGraph/) |[`final_flickr_separateGT_train.json`](https://huggingface.co/GLIPModel/GLIP/tree/main/mdetr_annotations/final_flickr_separateGT_train.json) |
| LVIS-minival | [`COCO val2017`](https://cocodataset.org/) | [`lvis_v1_minival_inserted_image_name.json`](https://huggingface.co/GLIPModel/GLIP/blob/main/lvis_v1_minival_inserted_image_name.json) |

**Acknowledgement:** The pre-training data preparation process is based on [YOLO-World](https://github.com/AILab-CVC/YOLO-World/blob/master/docs/data.md).


### Open-World Dataset

The data structure of `data/OWOD` is like

```bash
├── OWOD/
│   ├── JPEGImages/
│   │   ├── SOWODB/
│   │   ├── MOWODB/
│   │   └── nuOWODB/
│   ├── Annotations/
│   │   ├── SOWODB/
│   │   ├── MOWODB/
│   │   └── nuOWODB/
│   ├── ImageSets/
│   │   ├── SOWODB/
│   │   ├── MOWODB/
│   │   └── nuOWODB/
```

The splits and known texts prompt are present inside the `data/OWOD/ImageSets/MOWODB`, `data/OWOD/ImageSets/SOWODB` and `data/OWOD/ImageSets/nuOWODB` folders

- Download the COCO Images and Annotations from [MS-COCO](https://cocodataset.org/#download). Move all images from `train2017/` and `val2017/` to `JPEGImages` folder. Use the code `tools/dataset_converters/coco_to_voc.py` for converting json annotations to xml files.
- Download the [PASCAL VOC 2007 & 2012](http://host.robots.ox.ac.uk/pascal/VOC/) Images and Annotations. 
  Untar the trainval 2007 and 2012 and test 2007 folders. Move all the images to `JPEGImages` folder and annotations to `Annotations` folder.
- Download [nuImages](https://www.nuscenes.org/nuimages) for nuOWODB. Use the code `tools/dataset_converters/nuimages_to_voc.py` for converting both train/val annotations to xml files. 
  And move the related images and annotations to `JPEGImages` and `Annotations` folder. Make sure the filenames are matched with names in `ImageSets`.

**Note:** For M-OWODB/S-OWODB, we created just one folder of all the JPEG images and Annotations, for `SOWODB` and a symbolic link for `MOWODB`. We follow the VOC format for data loading and evaluation.


