import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.config import ConfigDict
from mmengine.model import BaseModule
from torch import Tensor
from torch.nn.modules.batchnorm import _BatchNorm

from mmengine.dist import get_dist_info
from mmdet.structures import SampleList
from mmdet.utils import InstanceList
from mmdet.models.utils import (multi_apply, unpack_gt_instances)
from mmyolo.registry import MODELS

from mmyolo.models.dense_heads import YOLOv10HeadModule, YOLOv10Head
from .yolo_world_head import ContrastiveHead, BNContrastiveHead


@MODELS.register_module()
class YOLOv10WorldHeadModule(YOLOv10HeadModule):
    """Head Module for YOLO-World

    Args:
        embed_dims (int): embed dim for text feautures and image features
        use_bn_head (bool): use batch normalization head
    """

    def __init__(self,
                 *args,
                 embed_dims: int,
                 use_bn_head: bool = False,
                 use_einsum: bool = True,
                 freeze_one2one: bool = False,
                 freeze_one2many: bool = False,
                 **kwargs) -> None:
        self.embed_dims = embed_dims
        self.use_bn_head = use_bn_head
        self.use_einsum = use_einsum
        self.freeze_one2one = freeze_one2one
        self.freeze_one2many = freeze_one2many
        super().__init__(*args, **kwargs)

    def init_weights(self, prior_prob=0.01):
        """Initialize the weight and bias of PPYOLOE head."""
        super().init_weights()
        for one2one_cls_pred, one2one_cls_contrast, \
        one2many_cls_pred, one2many_cls_contrast, stride in zip(self.one2one_cls_preds, 
                                                                self.one2one_cls_contrasts,
                                                                self.one2many_cls_preds,
                                                                self.one2many_cls_contrasts,
                                                                self.featmap_strides):
            one2one_cls_pred[-1].bias.data[:] = 0.0  # reset bias
            one2many_cls_pred[-1].bias.data[:] = 0.0  # reset bias
            if hasattr(one2one_cls_contrast, 'bias'):
                nn.init.constant_(
                    one2one_cls_contrast.bias.data,
                    math.log(5 / self.num_classes / (640 / stride)**2))
            if hasattr(one2many_cls_contrast, 'bias'):
                nn.init.constant_(
                    one2many_cls_contrast.bias.data,
                    math.log(5 / self.num_classes / (640 / stride)**2))

    def one2many_init_layers(self) -> None:
        """initialize conv layers in YOLOv8 head."""
        # Init decouple head
        self.one2many_cls_preds = nn.ModuleList()
        self.one2many_reg_preds = nn.ModuleList()
        self.one2many_cls_contrasts = nn.ModuleList()

        reg_out_channels = max((16, self.in_channels[0] // 4, self.reg_max * 4))
        cls_out_channels = max(self.in_channels[0], self.num_classes)
        c3 = max(self.in_channels[0], min(self.num_classes, 100))

        for i in range(self.num_levels):
            self.one2many_reg_preds.append(
                nn.Sequential(
                    ConvModule(
                        in_channels=self.in_channels[i],
                        out_channels=reg_out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg),
                    ConvModule(
                        in_channels=reg_out_channels,
                        out_channels=reg_out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg),
                    nn.Conv2d(
                        in_channels=reg_out_channels,
                        out_channels=4 * self.reg_max,
                        kernel_size=1)))
            self.one2many_cls_preds.append(
                nn.Sequential(
                        ConvModule(in_channels=self.in_channels[i],
                                out_channels=self.in_channels[i],
                                kernel_size=3,
                                stride=1,
                                groups=self.in_channels[i],
                                padding=1,
                                norm_cfg=self.norm_cfg,
                                act_cfg=self.act_cfg),
                        ConvModule(in_channels=self.in_channels[i],
                                out_channels=c3,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                norm_cfg=self.norm_cfg,
                                act_cfg=self.act_cfg), 
                        ConvModule(in_channels=c3,
                                out_channels=c3,
                                kernel_size=3,
                                groups=c3,
                                stride=1,
                                padding=1,
                                norm_cfg=self.norm_cfg,
                                act_cfg=self.act_cfg),
                        ConvModule(in_channels=c3,
                                out_channels=c3,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                norm_cfg=self.norm_cfg,
                                act_cfg=self.act_cfg),
                        nn.Conv2d(in_channels=c3,
                                out_channels=self.embed_dims,
                                kernel_size=1)
                    )
                )
            if self.use_bn_head:
                self.one2many_cls_contrasts.append(
                    BNContrastiveHead(self.embed_dims,
                                      self.norm_cfg,
                                      use_einsum=self.use_einsum))
            else:
                self.one2many_cls_contrasts.append(
                    ContrastiveHead(self.embed_dims,
                                    use_einsum=self.use_einsum))

        proj = torch.arange(self.reg_max, dtype=torch.float)
        self.register_buffer('one2many_proj', proj, persistent=False)

        if self.freeze_one2many:
            self._freeze_one2many()

    def one2one_init_layers(self) -> None:
        """initialize conv layers in YOLOv8 head."""
        # Init decouple head
        self.one2one_cls_preds = nn.ModuleList()
        self.one2one_reg_preds = nn.ModuleList()
        self.one2one_cls_contrasts = nn.ModuleList()

        reg_out_channels = max((16, self.in_channels[0] // 4, self.reg_max * 4))
        cls_out_channels = max(self.in_channels[0], self.num_classes)
        c3 = max(self.in_channels[0], min(self.num_classes, 100))

        for i in range(self.num_levels):
            self.one2one_reg_preds.append(
                nn.Sequential(
                    ConvModule(in_channels=self.in_channels[i],
                               out_channels=reg_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=self.norm_cfg,
                               act_cfg=self.act_cfg),
                    ConvModule(in_channels=reg_out_channels,
                               out_channels=reg_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=self.norm_cfg,
                               act_cfg=self.act_cfg),
                    nn.Conv2d(in_channels=reg_out_channels,
                              out_channels=4 * self.reg_max,
                              kernel_size=1)
                    )
                    )
            self.one2one_cls_preds.append(
                nn.Sequential(
                        ConvModule(in_channels=self.in_channels[i],
                                out_channels=self.in_channels[i],
                                kernel_size=3,
                                stride=1,
                                groups=self.in_channels[i],
                                padding=1,
                                norm_cfg=self.norm_cfg,
                                act_cfg=self.act_cfg),
                        ConvModule(in_channels=self.in_channels[i],
                                out_channels=c3,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                norm_cfg=self.norm_cfg,
                                act_cfg=self.act_cfg), 
                        ConvModule(in_channels=c3,
                                out_channels=c3,
                                kernel_size=3,
                                groups=c3,
                                stride=1,
                                padding=1,
                                norm_cfg=self.norm_cfg,
                                act_cfg=self.act_cfg),
                        ConvModule(in_channels=c3,
                                out_channels=c3,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                norm_cfg=self.norm_cfg,
                                act_cfg=self.act_cfg),
                        nn.Conv2d(in_channels=c3,
                                out_channels=self.embed_dims,
                                kernel_size=1)
                    )
                )
            if self.use_bn_head:
                self.one2one_cls_contrasts.append(
                    BNContrastiveHead(self.embed_dims,
                                      self.norm_cfg,
                                      use_einsum=self.use_einsum))
            else:
                self.one2one_cls_contrasts.append(
                    ContrastiveHead(self.embed_dims,
                                    use_einsum=self.use_einsum))

        proj = torch.arange(self.reg_max, dtype=torch.float)
        self.register_buffer('one2one_proj', proj, persistent=False)

        if self.freeze_one2one:
            self._freeze_one2one()

    def _freeze_one2many(self):
        """Freeze the model."""
        for n, m in self.named_modules():
            if 'one2many' in n:
                if isinstance(m, _BatchNorm):
                    m.eval()
                for param in m.parameters():
                    param.requires_grad = False
                    
    def _freeze_one2one(self):
        """Freeze the model."""
        for n, m in self.named_modules():
            if 'one2one' in n:
                if isinstance(m, _BatchNorm):
                    m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_one2one:
            self._freeze_one2one()
        if self.freeze_one2many:
            self._freeze_one2many()
            
    def forward_one2many(self, img_feats: Tuple[Tensor],
                         txt_feats: Tensor) -> Tuple[List]:
        """Forward features from the upstream network.

        Args:
            x (Tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            Tuple[List]: A tuple of multi-level classification scores, bbox
            predictions
        """
        assert len(img_feats) == self.num_levels
        txt_feats = [txt_feats for _ in range(self.num_levels)]
        return multi_apply(self.one2many_forward_single, img_feats, txt_feats, 
                           self.one2many_cls_preds, self.one2many_reg_preds, self.one2many_cls_contrasts)

    def forward_one2one(self, img_feats: Tuple[Tensor],
                         txt_feats: Tensor) -> Tuple[List]:
        """Forward features from the upstream network.

        Args:
            x (Tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            Tuple[List]: A tuple of multi-level classification scores, bbox
            predictions
        """
        assert len(img_feats) == self.num_levels
        txt_feats = [txt_feats for _ in range(self.num_levels)]
        return multi_apply(self.one2one_forward_single, img_feats, txt_feats,
                           self.one2one_cls_preds, self.one2one_reg_preds, self.one2one_cls_contrasts)

    def one2many_forward_single(self, img_feat: torch.Tensor, txt_feat: torch.Tensor,
                                cls_pred: nn.ModuleList, reg_pred: nn.ModuleList,
                                cls_contrast: nn.ModuleList) -> Tuple:
        """Forward feature of a single scale level."""
        b, _, h, w = img_feat.shape
        cls_embed = cls_pred(img_feat)
        cls_logit = cls_contrast(cls_embed, txt_feat)
        bbox_dist_preds = reg_pred(img_feat)
        
        if self.reg_max > 1:
            bbox_dist_preds = bbox_dist_preds.reshape([-1, 4, self.reg_max, h * w]).permute(0, 3, 1, 2)

            # TODO: The get_flops script cannot handle the situation of
            #  matmul, and needs to be fixed later
            # bbox_preds = bbox_dist_preds.softmax(3).matmul(self.proj)
            bbox_preds = bbox_dist_preds.softmax(3).matmul(self.one2many_proj.view([-1, 1])).squeeze(-1)
            bbox_preds = bbox_preds.transpose(1, 2).reshape(b, -1, h, w)
        else:
            bbox_preds = bbox_dist_preds
        if self.training:
            return cls_logit, bbox_preds, bbox_dist_preds
        else:
            return cls_logit, bbox_preds

    def one2one_forward_single(self, img_feat: torch.Tensor, txt_feat: torch.Tensor,
                                cls_pred: nn.ModuleList, reg_pred: nn.ModuleList,
                                cls_contrast: nn.ModuleList) -> Tuple:
        """Forward feature of a single scale level."""
        b, _, h, w = img_feat.shape
        cls_embed = cls_pred(img_feat)
        cls_logit = cls_contrast(cls_embed, txt_feat) #bkhw

        # cls_logit[:, self.num_classes-1:self.num_classes] = torch.amax(cls_logit[:, self.num_classes-1:], dim=1, keepdim=True)
        # cls_logit = cls_logit[:, :self.num_classes]

        bbox_dist_preds = reg_pred(img_feat)

        if self.reg_max > 1:
            bbox_dist_preds = bbox_dist_preds.reshape([-1, 4, self.reg_max, h * w]).permute(0, 3, 1, 2)

            # TODO: The get_flops script cannot handle the situation of
            #  matmul, and needs to be fixed later
            # bbox_preds = bbox_dist_preds.softmax(3).matmul(self.proj)
            bbox_preds = bbox_dist_preds.softmax(3).matmul(self.one2one_proj.view([-1, 1])).squeeze(-1)
            bbox_preds = bbox_preds.transpose(1, 2).reshape(b, -1, h, w)
        else:
            bbox_preds = bbox_dist_preds
        
        if self.training:
            return cls_logit, bbox_preds, bbox_dist_preds
        else:
            return cls_logit, bbox_preds


@MODELS.register_module()
class YOLOv10WorldHead(YOLOv10Head):
    """YOLO-World Head
    """

    def __init__(self, 
                 world_size=-1,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.world_size = world_size

    """YOLO World v10 head."""

    def loss(self, img_feats: Tuple[Tensor], txt_feats: Tensor,
             batch_data_samples: Union[list, dict]) -> dict:
        """Perform forward propagation and loss calculation of the detection
        head on the features of the upstream network."""

        one2many_outs, one2one_outs = self(img_feats, txt_feats)

        if isinstance(batch_data_samples, list):
            losses = super(YOLOv10Head, self).loss(one2many_outs, batch_data_samples)
        else:
            # Fast version
            one2many_loss_inputs = one2many_outs + (batch_data_samples['bboxes_labels'], batch_data_samples['img_metas'])
            one2one_loss_inputs = one2one_outs + (batch_data_samples['bboxes_labels'], batch_data_samples['img_metas'])
            losses = self.loss_by_feat([one2many_loss_inputs, one2one_loss_inputs])

        return losses

    def loss_and_predict(
        self,
        img_feats: Tuple[Tensor],
        txt_feats: Tensor,
        batch_data_samples: SampleList,
        proposal_cfg: Optional[ConfigDict] = None
    ) -> Tuple[dict, InstanceList]:
        """Perform forward propagation of the head, then calculate loss and
        predictions from the features and data samples.
        """
        outputs = unpack_gt_instances(batch_data_samples)
        (batch_gt_instances, batch_gt_instances_ignore,
         batch_img_metas) = outputs

        one2many_outs, one2one_outs = self(img_feats, txt_feats)
        
        one2many_loss_inputs = one2many_outs + (batch_gt_instances, batch_img_metas,
                            batch_gt_instances_ignore)
        one2one_loss_inputs = one2one_outs + (batch_gt_instances, batch_img_metas,
                            batch_gt_instances_ignore)
        losses = self.loss_by_feat([one2many_loss_inputs, one2one_loss_inputs])

        if self.infer_type == "one2one":
            outs = one2one_outs
            with_nms = self.test_cfg.get("one2one_withnms", False)
        elif self.infer_type == "one2many":
            outs = one2many_outs
            with_nms = self.test_cfg.get("one2many_withnms", False)
        else:
            raise Exception("unsupported infer type")

        predictions = self.predict_by_feat(*outs,
                                            batch_img_metas=batch_img_metas,
                                            with_nms=with_nms)

        return losses, predictions

    def forward(self, img_feats: Tuple[Tensor],
                txt_feats: Tensor) -> Tuple[List]:
        """Forward features from the upstream network."""
        if self.training:
            one2many_result = self.head_module.forward_one2many(img_feats, txt_feats)
            img_feats_detach = [img_feat.detach() for img_feat in img_feats]
            one2one_result = self.head_module.forward_one2one(img_feats_detach, txt_feats)
        else:
            one2many_result = self.head_module.forward_one2many(img_feats, txt_feats) if self.infer_type == 'one2many' else None
            one2one_result = self.head_module.forward_one2one(img_feats, txt_feats) if self.infer_type == 'one2one' else None

        return one2many_result, one2one_result

    def predict(self,
                img_feats: Tuple[Tensor],
                txt_feats: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = False) -> InstanceList:
        """Perform forward propagation of the detection head and predict
        detection results on the features of the upstream network.
        """
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        one2many_outs, one2one_outs = self(img_feats, txt_feats)

        if self.infer_type == "one2one":
            outs = one2one_outs
            with_nms = self.test_cfg.get("one2one_withnms", False)
        elif self.infer_type == "one2many":
            outs = one2many_outs
            with_nms = self.test_cfg.get("one2many_withnms", False)
        else:
            raise Exception("unsupported infer type")

        predictions = self.predict_by_feat(*outs,
                                            batch_img_metas=batch_img_metas,
                                            rescale=rescale,
                                            with_nms=with_nms)
        return predictions

    def aug_test(self,
                 aug_batch_feats,
                 aug_batch_img_metas,
                 rescale=False,
                 with_ori_nms=False,
                 **kwargs):
        """Test function with test time augmentation."""
        raise NotImplementedError('aug_test is not implemented yet.')

    def loss_by_feat(self, all2one) -> dict:
        one2many_loss_inputs = all2one[0]
        one2one_loss_inputs = all2one[1]

        losses = self.one2many_loss_by_feat(*one2many_loss_inputs)
        losses.update(self.one2one_loss_by_feat(*one2one_loss_inputs)) 
        return losses