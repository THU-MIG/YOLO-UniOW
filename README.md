# [YOLO-UniOW: Efficient Universal Open-World Object Detection]()

![yolo-uniow](./assets/yolo-uniow.jpg)

[YOLO-UniOW: Efficient Universal Open-World Object Detection]()

Lihao Liu, Juexiao Feng, Hui Chen, Ao Wang, Lin Song, Guiguang Ding

## Zero-shot Performance On LVIS Dataset

| Model                                                        | #Params | $\mathrm{AP^{mini}}$ | $\mathrm{AP_r}$ | $\mathrm{AP_c}$ | $\mathrm{AP_f}$ | FPS (V100) |
| ------------------------------------------------------------ | ------- | -------------------- | --------------- | --------------- | --------------- | ---------- |
| [YOLO-UniOW-S](https://huggingface.co/leonnil/yolo-uniow/resolve/main/yolo_uniow_s_lora_bn_5e-4_100e_8gpus_obj365v1_goldg_train_lvis_minival.pth) | 7.5M    | 26.2                 | 24.1            | 24.9            | 27.7            | 119.3       |
| [YOLO-UniOW-M](https://huggingface.co/leonnil/yolo-uniow/resolve/main/yolo_uniow_m_lora_bn_5e-4_100e_8gpus_obj365v1_goldg_train_lvis_minival.pth) | 16.2M   | 31.8                 | 26.0            | 30.5            | 34              | 98.9       |
| [YOLO-UniOW-L](https://huggingface.co/leonnil/yolo-uniow/resolve/main/yolo_uniow_l_lora_bn_5e-4_100e_8gpus_obj365v1_goldg_train_lvis_minival.pth) | 29.4M   | 34.6                 | 30.0            | 33.6            | 36.3            | 69.6       |

## Experiment Setup

### Data Preparation

For open-vocabulary and open-world data, please refer to [docs/data](./docs/data.md)

### Installation

Our model is developed using **CUDA 11.8** and **PyTorch 2.1.2**. To set up the environment, please refer to the [PyTorch official documentation](https://pytorch.org/get-started/locally/) for installation instructions.  

**Important:** Before running `pip install -r requirements.txt`, please refer to [docs/installation](./docs/installation.md) for specific instructions on installing `mmcv`.

```bash
conda create -n yolouniow python=3.9
conda activate yolouniow
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
pip install -r requirements.txt
pip install -e .
```

### Training & Evaluation

For Open-Vocabulary model training and evaluation, please refer to `run_ovod.sh`

```bash
# Train Open-Vocabulary Model
./tools/dist_train.sh configs/pretrain/yolo_uniow_s_lora_bn_5e-4_100e_8gpus_obj365v1_goldg_train_lvis_minival.py 8 --amp

# Evaluate Open-Vocabulary Model
./tools/dist_test.sh configs/pretrain/yolo_uniow_s_lora_bn_5e-4_100e_8gpus_obj365v1_goldg_train_lvis_minival.py \
    pretrained/yolo_uniow_s_lora_bn_5e-4_100e_8gpus_obj365v1_goldg_train_lvis_minival.pth 8
```

For Open-World model training and evaluation, please follow the steps provided in `run_owod.sh`. Ensure that the model is trained before proceeding with the evaluation.

```bash
# 1. Extract text/wildcard features from pretrained model
python tools/owod_scripts/extract_text_feats.py --config $CONFIG --ckpt $CHECKPOINT --save_path $EMBEDS_PATH

# 2. Fine-tune wildcard features
./tools/dist_train.sh $OBJ_CONFIG 8 --amp

# 3. Extract fine-tuned wildcard features
python tools/owod_scripts/extract_text_feats.py --config $OBJ_CONFIG --save_path $EMBEDS_PATH --extract_tuned

# 4. Train all owod tasks
python tools/owod_scripts/train_owod_tasks.py MOWODB $OW_CONFIG $CHECKPOINT

# 5. Evaluate all owod tasks
python tools/owod_scripts/test_owod_tasks.py MOWODB $OW_CONFIG --save
```

For running specific dataset and task, you can follow:

```bash
DATASET=$DATASET TASK=$TASK THRESHOLD=$THRESHOLD SAVE=$SAVE \
./tools/dist_train_owod.sh $CONFIG 8 --amp

DATASET=$DATASET TASK=$TASK THRESHOLD=$THRESHOLD SAVE=$SAVE \
./tools/dist_test.sh $CONFIG $CHECKPOINT 8
```

## Acknowledgement

The code and data are built based on [YOLO-World](https://github.com/AILab-CVC/YOLO-World),  [YOLOv10](https://github.com/Trami1995/YOLOv10), [FOMO](https://github.com/orrzohar/FOMO) and [OVOW](https://github.com/343gltysprk/ovow/). Thanks for their great implementations!
