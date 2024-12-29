#!/usr/bin/bash

chmod +x tools/dist_train.sh
chmod +x tools/dist_test.sh

# Train Open-Vocabulary Model
./tools/dist_train.sh configs/pretrain/yolo_uniow_s_lora_bn_5e-4_100e_8gpus_obj365v1_goldg_train_lvis_minival.py 8 --amp
./tools/dist_train.sh configs/pretrain/yolo_uniow_m_lora_bn_5e-4_100e_8gpus_obj365v1_goldg_train_lvis_minival.py 8 --amp
./tools/dist_train.sh configs/pretrain/yolo_uniow_l_lora_bn_5e-4_100e_8gpus_obj365v1_goldg_train_lvis_minival.py 8 --amp

# Evaluate Open-Vocabulary Model
./tools/dist_test.sh configs/pretrain/yolo_uniow_s_lora_bn_5e-4_100e_8gpus_obj365v1_goldg_train_lvis_minival.py \
    pretrained/yolo_uniow_s_lora_bn_5e-4_100e_8gpus_obj365v1_goldg_train_lvis_minival.pth 8

./tools/dist_test.sh configs/pretrain/yolo_uniow_m_lora_bn_5e-4_100e_8gpus_obj365v1_goldg_train_lvis_minival.py \
    pretrained/yolo_uniow_m_lora_bn_5e-4_100e_8gpus_obj365v1_goldg_train_lvis_minival.pth 8

./tools/dist_test.sh configs/pretrain/yolo_uniow_l_lora_bn_5e-4_100e_8gpus_obj365v1_goldg_train_lvis_minival.py \
    pretrained/yolo_uniow_l_lora_bn_5e-4_100e_8gpus_obj365v1_goldg_train_lvis_minival.pth 8
