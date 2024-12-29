_base_ = [('../../third_party/mmyolo/configs/yolov10/'
          'yolov10_m_syncbn_fast_8xb16-500e_coco.py')]
custom_imports = dict(imports=['yolo_world'],
                      allow_failed_imports=False)

# hyper-parameters
num_classes = 1
num_training_classes = 1
max_epochs = 3  # Maximum training epochs
close_mosaic_epochs = 1
save_epoch_intervals = 1
text_channels = 512
neck_embed_channels = [128, 256, _base_.last_stage_out_channels // 2]
neck_num_heads = [4, 8, _base_.last_stage_out_channels // 2 // 32]
base_lr = 1e-4
weight_decay = 0.025
train_batch_size_per_gpu = 16

load_from = 'pretrained/yolo_uniow_m_lora_bn_5e-4_100e_8gpus_obj365v1_goldg_train_lvis_minival.pth'

# model settings
infer_type = "one2one"

model = dict(
    type='OWODDetector',
    mm_neck=False,
    num_train_classes=num_training_classes,
    num_test_classes=num_classes,
    num_prompts=num_classes,
    freeze_prompt=False,
    embedding_path='embeddings/uniow-m/object.npy',
    data_preprocessor=dict(type='YOLOv5DetDataPreprocessor'),
    backbone=dict(
        _delete_=True,
        type='MultiModalYOLOBackbone',
        image_model={{_base_.model.backbone}},
        text_model=None,
        with_text_model=False,
        frozen_stages=4,
    ),
    neck=dict(
        freeze_all=True,
    ),
    bbox_head=dict(type='YOLOv10WorldHead',
                   infer_type=infer_type,
                   head_module=dict(type='YOLOv10WorldHeadModule',
                                    use_bn_head=True,
                                    freeze_one2one=True,
                                    freeze_one2many=True,
                                    embed_dims=text_channels,
                                    num_classes=num_training_classes)),
    train_cfg=dict(
        one2many_assigner=dict(num_classes=num_training_classes),
        one2one_assigner=dict(num_classes=num_training_classes)),
)


# dataset settings
text_transform = [
    dict(type='ClassAgnosticLabel'),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                    'flip_direction'))
]
train_pipeline = [
    *_base_.train_pipeline[:-1],
    *text_transform,
]

train_pipeline_stage2 = [*_base_.train_pipeline_stage2[:-1], *text_transform]
obj365v1_train_dataset = dict(
    type='MultiModalDataset',
    dataset=dict(
        type='YOLOv5Objects365V1Dataset',
        data_root='data/objects365v1/',
        ann_file='annotations/objects365_train.json',
        data_prefix=dict(img='train/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path='data/texts/obj365v1_class_texts.json',
    pipeline=train_pipeline)

mg_train_dataset = dict(type='YOLOv5MixedGroundingDataset',
                        data_root='data/mixed_grounding/',
                        ann_file='annotations/final_mixed_train_no_coco.json',
                        data_prefix=dict(img='gqa/images/'),
                        filter_cfg=dict(filter_empty_gt=False, min_size=32),
                        pipeline=train_pipeline)

flickr_train_dataset = dict(
    type='YOLOv5MixedGroundingDataset',
    data_root='data/flickr/',
    ann_file='annotations/final_flickr_separateGT_train.json',
    data_prefix=dict(img='full_images/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=train_pipeline)

train_dataloader = dict(batch_size=train_batch_size_per_gpu,
                        collate_fn=dict(type='yolow_collate'),
                        dataset=dict(_delete_=True,
                                     type='ConcatDataset',
                                     datasets=[
                                         obj365v1_train_dataset,
                                         flickr_train_dataset,
                                         mg_train_dataset,
                                     ],
                                     ignore_keys=['classes', 'palette']))

test_pipeline = [
    *_base_.test_pipeline[:-1],
    dict(type='ClassAgnosticLabel'),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'pad_param'))
]
coco_val_dataset = dict(
    _delete_=True,
    type='MultiModalDataset',
    dataset=dict(type='YOLOv5LVISV1Dataset',
                 data_root='data/coco/',
                 test_mode=True,
                 ann_file='lvis/lvis_v1_minival_inserted_image_name.json',
                 data_prefix=dict(img=''),
                 batch_shapes_cfg=None),
    class_text_path='data/texts/lvis_v1_class_texts.json',
    pipeline=test_pipeline)
val_dataloader = dict(dataset=coco_val_dataset)
test_dataloader = val_dataloader

val_evaluator = dict(type='mmdet.LVISMetric',
                     ann_file='data/coco/lvis/lvis_v1_minival_inserted_image_name.json',
                     proposal_nums=(100, 300, 1000),
                     metric='proposal_fast')
test_evaluator = val_evaluator

# training settings
default_hooks = dict(param_scheduler=dict(max_epochs=max_epochs),
                     checkpoint=dict(interval=save_epoch_intervals,
                                     rule='greater'))
custom_hooks = [
    dict(type='EMAHook',
         ema_type='ExpMomentumEMA',
         momentum=0.0001,
         update_buffers=True,
         strict_load=False,
         priority=49),
    dict(type='mmdet.PipelineSwitchHook',
         switch_epoch=max_epochs - close_mosaic_epochs,
         switch_pipeline=train_pipeline_stage2)
]
train_cfg = dict(max_epochs=max_epochs,
                 val_interval=1,
                 dynamic_intervals=[((max_epochs - close_mosaic_epochs),
                                     _base_.val_interval_stage2)])
optim_wrapper = dict(optimizer=dict(
    _delete_=True,
    type='AdamW',
    lr=base_lr,
    weight_decay=weight_decay,
    batch_size_per_gpu=train_batch_size_per_gpu),
                     paramwise_cfg=dict(bias_decay_mult=0.0,
                                        norm_decay_mult=0.0,
                                        custom_keys={
                                            'backbone.text_model':
                                            dict(lr_mult=0.01),
                                            'logit_scale':
                                            dict(weight_decay=0.0),
                                            'embeddings':
                                            dict(weight_decay=0.0)
                                        }),
                     constructor='YOLOWv5OptimizerConstructor')