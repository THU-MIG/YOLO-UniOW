#!/usr/bin/bash

set -e

CONFIG=configs/pretrain/yolo_uniow_s_lora_bn_5e-4_100e_8gpus_obj365v1_goldg_train_lvis_minival.py
CHECKPOINT=pretrained/yolo_uniow_s_lora_bn_5e-4_100e_8gpus_obj365v1_goldg_train_lvis_minival.pth

OW_CONFIG=configs/owod_ft/yolo_uniow_s_lora_bn_1e-3_20e_8gpus_owod.py
OBJ_CONFIG=configs/owod_ft/yolo_uniow_s_lora_object_ft_bn_1e-4_3e_8gpus_owod.py

EMBEDS_PATH=embeddings/uniow-s

# 1. Extract text/wildcard features from pretrained model
python tools/owod_scripts/extract_text_feats.py --config $CONFIG --ckpt $CHECKPOINT --save_path $EMBEDS_PATH

# 2. Fine-tune wildcard features
./tools/dist_train.sh $OBJ_CONFIG 8 --amp

# 3. Extract fine-tuned wildcard features
python tools/owod_scripts/extract_text_feats.py --config $OBJ_CONFIG --save_path $EMBEDS_PATH --extract_tuned

# 4. Train all owod tasks
python tools/owod_scripts/train_owod_tasks.py MOWODB $OW_CONFIG $CHECKPOINT
python tools/owod_scripts/train_owod_tasks.py SOWODB $OW_CONFIG $CHECKPOINT
python tools/owod_scripts/train_owod_tasks.py nuOWODB $OW_CONFIG $CHECKPOINT

# 5. Evaluate all owod tasks
python tools/owod_scripts/test_owod_tasks.py MOWODB $OW_CONFIG --save
python tools/owod_scripts/test_owod_tasks.py SOWODB $OW_CONFIG --save
python tools/owod_scripts/test_owod_tasks.py nuOWODB $OW_CONFIG --save