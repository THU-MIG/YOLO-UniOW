# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Union

import torch
import torch.nn as nn
from mmdet.utils import ConfigType, OptMultiConfig
from mmcv.cnn import ConvModule

from mmyolo.registry import MODELS
from mmengine.model import constant_init, kaiming_init

from .. import CSPLayerWithTwoConv, SCDown, C2fCIB
from ..utils import make_divisible, make_round
from .yolov5_pafpn import YOLOv5PAFPN


@MODELS.register_module()
class YOLOv10PAFPN(YOLOv5PAFPN):
    """Path Aggregation Network used in YOLOv10.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        deepen_factor (float): Depth multiplier, multiply number of
            blocks in CSP layer by this amount. Defaults to 1.0.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Defaults to 1.0.
        num_csp_blocks (int): Number of bottlenecks in CSPLayer. Defaults to 1.
        freeze_all(bool): Whether to freeze the model
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='SiLU', inplace=True).
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: List[int],
                 out_channels: Union[List[int], int],
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 num_csp_blocks: int = 3,
                 freeze_all: bool = False,
                 lk: bool = False,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None):
        self.lk = lk

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            deepen_factor=deepen_factor,
            widen_factor=widen_factor,
            num_csp_blocks=num_csp_blocks,
            freeze_all=freeze_all,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            init_cfg=init_cfg)

    def build_upsample_layer(self, *args, **kwargs) -> nn.Module:
        """build upsample layer."""
        return nn.Upsample(scale_factor=2, mode='nearest')

    def build_downsample_layer(self, idx: int) -> nn.Module:
        """build downsample layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The downsample layer.
        """
        if idx > 0:
            return SCDown(
                make_divisible(self.in_channels[idx], self.widen_factor),
                make_divisible(self.in_channels[idx], self.widen_factor),
                3, 2)
        else:
            return ConvModule(
                make_divisible(self.in_channels[idx], self.widen_factor),
                make_divisible(self.in_channels[idx], self.widen_factor),
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)

    def build_reduce_layer(self, idx: int) -> nn.Module:
        """build reduce layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The reduce layer.
        """
        return nn.Identity()

    def build_top_down_layer(self, idx: int) -> nn.Module:
        """build top down layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The top down layer.
        """
        return CSPLayerWithTwoConv(
            make_divisible((self.in_channels[idx - 1] + self.in_channels[idx]),
                           self.widen_factor),
            make_divisible(self.out_channels[idx - 1], self.widen_factor),
            num_blocks=make_round(self.num_csp_blocks, self.deepen_factor),
            add_identity=False,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def build_bottom_up_layer(self, idx: int) -> nn.Module:
        """build bottom up layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The bottom up layer.
        """
        if idx > 0:
            return C2fCIB(
                make_divisible(
                    (self.out_channels[idx] + self.out_channels[idx + 1]),
                    self.widen_factor),
                make_divisible(self.out_channels[idx + 1], self.widen_factor),
                shortcut=True,
                lk=self.lk,
            )
        else:
            return CSPLayerWithTwoConv(
                make_divisible(
                    (self.out_channels[idx] + self.out_channels[idx + 1]),
                    self.widen_factor),
                make_divisible(self.out_channels[idx + 1], self.widen_factor),
                num_blocks=make_round(self.num_csp_blocks, self.deepen_factor),
                add_identity=False,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m, mode="fan_out",
                                nonlinearity='relu',
                                distribution='normal')  # leaky_relu
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                constant_init(m, 1, bias=0)

    def forward(self, inputs: List[torch.Tensor]) -> tuple:
        """Forward function."""
        assert len(inputs) == len(self.in_channels)
        # reduce layers
        reduce_outs = []
        for idx in range(len(self.in_channels)):
            reduce_outs.append(self.reduce_layers[idx](inputs[idx]))

        # top-down path
        inner_outs = [reduce_outs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = reduce_outs[idx - 1]
            upsample_feat = self.upsample_layers[len(self.in_channels) - 1 -
                                                 idx](
                                                     feat_high)
            if self.upsample_feats_cat_first:
                top_down_layer_inputs = torch.cat([upsample_feat, feat_low], 1)
            else:
                top_down_layer_inputs = torch.cat([feat_low, upsample_feat], 1)
            inner_out = self.top_down_layers[len(self.in_channels) - 1 - idx](
                top_down_layer_inputs)
            inner_outs.insert(0, inner_out)

        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_layers[idx](feat_low)
            out = self.bottom_up_layers[idx](
                torch.cat([downsample_feat, feat_high], 1))
            outs.append(out)

        # out_layers
        results = []
        for idx in range(len(self.in_channels)):
            results.append(self.out_layers[idx](outs[idx]))

        return tuple(results)