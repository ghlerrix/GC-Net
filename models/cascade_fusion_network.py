# Copyright (c) Ghlerrix. All rights reserved.
from typing import List, Union
import torch
from abc import ABCMeta
import torch.nn as nn
from mmdet.utils import ConfigType, OptMultiConfig
from mmcv.cnn import ConvModule
from mmyolo.registry import MODELS
from mmyolo.models import CSPLayerWithTwoConv
from mmyolo.models.utils import make_divisible, make_round
from mmengine.model import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm


@MODELS.register_module()
class CFN(BaseModule, metaclass=ABCMeta):
    def __init__(self,
                 num_cascade: int,
                 in_channels: List[int],
                 out_channels: Union[List[int], int],
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 num_csp_blocks: int = 3,
                 freeze_all: bool = False,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.deepen_factor = deepen_factor
        self.widen_factor = widen_factor
        self.num_csp_blocks = num_csp_blocks
        self.freeze_all = freeze_all
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.num_cascade = num_cascade

        self.upsample_layer = nn.Upsample(scale_factor=2, mode='nearest')

        self.top_down_layers = nn.ModuleList()
        for _ in range(num_cascade + 1):
            top_down_layer = nn.ModuleList()
            for idx in range(len(self.in_channels) - 1, 0, -1):
                top_down_layer.append(
                    CSPLayerWithTwoConv(
                        make_divisible(
                            (self.in_channels[idx - 1] +
                             self.in_channels[idx]),
                            self.widen_factor),
                        make_divisible(
                            self.out_channels[idx - 1],
                            self.widen_factor),
                        num_blocks=make_round(
                            self.num_csp_blocks, self.deepen_factor),
                        add_identity=False,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg)
                )
            self.top_down_layers.append(top_down_layer)

        self.downsample_layers = nn.ModuleList()
        for _ in range(num_cascade):
            downsample_layer = nn.ModuleList()
            for idx in range(0, len(self.in_channels) - 1):
                downsample_layer.append(
                    ConvModule(
                        make_divisible(
                            self.in_channels[idx], self.widen_factor),
                        make_divisible(
                            self.in_channels[idx + 1], self.widen_factor),
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
            self.downsample_layers.append(downsample_layer)

    def _freeze_all(self):
        """Freeze the model."""
        for m in self.modules():
            if isinstance(m, _BatchNorm):
                m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        """Convert the model into training mode while keep the normalization
        layer freezed."""
        super().train(mode)
        if self.freeze_all:
            self._freeze_all()

    def forward(self, inputs: List[torch.Tensor]) -> tuple:
        assert len(inputs) == len(self.in_channels)

        feat_high = inputs[-1]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_low = inputs[idx - 1]
            upsample_feat = self.upsample_layer(feat_high)
            top_down_layer_inputs = torch.cat([upsample_feat, feat_low], dim=1)
            feat_high = self.top_down_layers[0][len(self.in_channels) - 1 -
                                                idx](top_down_layer_inputs)

        for i in range(0, self.num_cascade - 1):
            inner_outs = [feat_high]
            for idx in range(0, len(self.in_channels) - 1):
                out = self.downsample_layers[i][idx](inner_outs[-1])
                inner_outs.append(out)

            feat_high = inner_outs[-1]
            for idx in range(len(self.in_channels) - 1, 0, -1):
                feat_low = inner_outs[idx - 1]
                upsample_feat = self.upsample_layer(feat_high)
                top_down_layer_inputs = torch.cat([upsample_feat, feat_low], dim=1)
                feat_high = self.top_down_layers[i + 1][len(self.in_channels) - 1 -
                                                    idx](top_down_layer_inputs)
            
        outs = [feat_high]
        for idx in range(0, len(self.in_channels) - 1):
            out = self.downsample_layers[-1][idx](outs[-1])
            outs.append(out)

        feat_high = outs[-1]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_low = outs[idx - 1]
            upsample_feat = self.upsample_layer(feat_high)
            top_down_layer_inputs = torch.cat([upsample_feat, feat_low], dim=1)
            feat_high = self.top_down_layers[-1][len(self.in_channels) - 1 -
                                                idx](top_down_layer_inputs)

        results = [feat_high]
        for idx in range(1, len(self.in_channels)):
            results.append(outs[idx])

        return tuple(results)
