from typing import List, Tuple, Union

import torch.nn as nn

from mmcv.cnn import ConvModule
from ..layers import CSPLayerWithTwoConv, SPPF, SCDown, C2fCIB, PSA
from ..utils import make_divisible, make_round
from mmyolo.registry import MODELS
from .base_backbone import BaseBackbone
from mmdet.utils import ConfigType, OptMultiConfig
from mmengine.model import constant_init, kaiming_init


@MODELS.register_module()
class YOLOv10Backbone(BaseBackbone):
    arch_settings = {
        'P5': [[64, 128, 3, True, False, False, False, False], [128, 256, 6, True, False, False, False, False],
               [256, 512, 6, True, True, False, False, False], [512, None, 3, True, True, None, True, False]],
    }

    def __init__(self, 
                 arch: str = 'P5',
                 last_stage_out_channels: int = 1024,
                 use_c2fcib: bool = True,
                 lk: bool = False,
                 plugins: Union[dict, List[dict]] = None,
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 input_channels: int = 3,
                 out_indices: Tuple[int] = (2, 3, 4),
                 frozen_stages: int = -1,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 norm_eval: bool = False,
                 init_cfg: OptMultiConfig = None):
        self.arch_settings[arch][-1][1] = last_stage_out_channels
        self.arch_settings[arch][-1][5] = use_c2fcib
        self.arch_settings[arch][-1][-1] = lk
        super().__init__(
            self.arch_settings[arch],
            deepen_factor,
            widen_factor,
            input_channels=input_channels,
            out_indices=out_indices,
            plugins=plugins,
            frozen_stages=frozen_stages,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            norm_eval=norm_eval,
            init_cfg=init_cfg)

    def build_stem_layer(self):
        """Build a stem layer."""
        return ConvModule(
            self.input_channels,
            make_divisible(self.arch_setting[0][0], self.widen_factor),
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def build_stage_layer(self, stage_idx: int, setting: list):
        """Build a stage layer.

        Args:
            stage_idx (int): The index of a stage layer.
            setting (list): The architecture setting of a stage layer.
        """
        in_channels, out_channels, num_blocks, add_identity, use_scdown, use_c2fcib, use_psa, lk = setting

        in_channels = make_divisible(in_channels, self.widen_factor)
        out_channels = make_divisible(out_channels, self.widen_factor)
        num_blocks = make_round(num_blocks, self.deepen_factor)
        stage = []
        if use_scdown:
            sc_down_layer = SCDown(in_channels, 
                out_channels,
                k=3, 
                s=2)
            stage.append(sc_down_layer)
        else:
            conv_layer = ConvModule(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
            stage.append(conv_layer)

        if use_c2fcib:
            c2fcib_layer = C2fCIB(out_channels, 
                out_channels, 
                n=num_blocks, 
                shortcut=add_identity,
                lk=lk)
            stage.append(c2fcib_layer)
        else:
            csp_layer = CSPLayerWithTwoConv(
                out_channels,
                out_channels,
                num_blocks=num_blocks,
                add_identity=add_identity,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
            stage.append(csp_layer)

        if use_psa:
            sppf_layer = SPPF(out_channels, 
                out_channels, 
                k=5)
            psa_layer = PSA(out_channels, 
                out_channels)
            stage.append(sppf_layer)
            stage.append(psa_layer)
            
        return stage

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m, mode="fan_out",
                                nonlinearity='relu',
                                distribution='normal')  # leaky_relu
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                constant_init(m, 1, bias=0)
