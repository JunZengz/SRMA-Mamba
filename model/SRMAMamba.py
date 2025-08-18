from __future__ import annotations

from typing import Union

import torch.nn as nn
import torch

from monai.networks.blocks.dynunet_block import UnetOutBlock
import torch.nn.functional as F
from .vmamba2 import SABMambaEncoder, LayerNorm3d, Permute, SABMamba

from collections.abc import Sequence
from monai.networks.blocks.dynunet_block import UnetBasicBlock, UnetResBlock, get_conv_layer

class UnetrBasicBlock(nn.Module):
    """
    A CNN module that can be used for UNETR, based on: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        norm_name: Union[tuple, str],
        res_block: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            stride: convolution stride.
            norm_name: feature normalization type and arguments.
            res_block: bool argument to determine if residual block is used.

        """

        super().__init__()

        if res_block:
            self.layer = UnetResBlock(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                norm_name=norm_name,
            )
        else:
            self.layer = UnetBasicBlock(  # type: ignore
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                norm_name=norm_name,
            )

    def forward(self, inp):
        return self.layer(inp)


class SRMAttention(nn.Module):
    def __init__(self, channel, channel_first, scale,**kwargs):
        super(SRMAttention, self).__init__()

        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm3d,
            bn=nn.BatchNorm3d,
        )

        _ACTLAYERS = dict(
            silu=nn.SiLU,
            gelu=nn.GELU,
            relu=nn.ReLU,
            sigmoid=nn.Sigmoid,
        )

        norm_layer: nn.Module = _NORMLAYERS.get(kwargs['norm_layer'].lower(), None)
        ssm_act_layer: nn.Module = _ACTLAYERS.get(kwargs['ssm_act_layer'].lower(), None)
        mlp_act_layer: nn.Module = _ACTLAYERS.get(kwargs['mlp_act_layer'].lower(), None)

        # self.cur_volume_shape = tuple(dim // scale for dim in kwargs['volume_shape'])

        self.st_block1 = nn.Sequential(
            Permute(0, 2, 3, 4, 1) if not channel_first else nn.Identity(),
            SABMamba(hidden_dim=channel, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 4, 1, 2, 3) if not channel_first else nn.Identity(),
        )
        self.st_block2 = nn.Sequential(
            Permute(0, 2, 3, 4, 1) if not channel_first else nn.Identity(),
            SABMamba(hidden_dim=channel, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 4, 1, 2, 3) if not channel_first else nn.Identity(),
        )
        self.ra_conv1 =  UnetOutBlock(spatial_dims=3, in_channels=channel, out_channels=channel)
        self.ra_conv2 = UnetOutBlock(spatial_dims=3, in_channels=channel, out_channels=channel)
        self.ra_conv3 = UnetOutBlock(spatial_dims=3, in_channels=channel, out_channels=1)
        self.res = lambda x, size: F.interpolate(x, size=size, mode='trilinear', align_corners=False)


    def forward(self, x, map):
        b, c, h, w, d = x.shape
        base_size = (h, w, d)
        map = self.res(map, base_size)
        attn = -1*(torch.sigmoid(map)) + 1

        identity = x
        x = self.st_block1(x)
        x = attn.expand(-1, c, -1, -1, -1).mul(x)
        x = self.ra_conv1(x)
        x = self.st_block2(x)
        x += identity
        x = self.ra_conv2(x)
        x = self.ra_conv3(x)
        res = x + map
        return res


class BasicConv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm3d(out_planes)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class SRMAMamba(nn.Module):
    def __init__(
            self,
            in_chans=1,
            out_chans=1,
            embed_channels=16,
            drop_path_rate=0,
            layer_scale_init_value=1e-6,
            hidden_size: int = 768,
            norm_name="instance",
            conv_block: bool = True,
            res_block: bool = True,
            spatial_dims=3,
            **kwargs,
    ) -> None:
        super().__init__()

        # self.volume_shape = kwargs['volume_shape']
        self.hidden_size = hidden_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        # self.depths = depths
        self.drop_path_rate = drop_path_rate
        self.feats_size = [kwargs['dims'] * (2**i) for i in range(4)]
        self.layer_scale_init_value = layer_scale_init_value

        self.spatial_dims = spatial_dims
        self.encoder = SABMambaEncoder(out_indices=(0, 1, 2, 3), **kwargs)
        self.channel_first = self.encoder.channel_first

        self.translayer1_st = BasicConv3d(self.feats_size[0], embed_channels, 1)
        self.translayer2_st = BasicConv3d(self.feats_size[1], embed_channels, 1)
        self.translayer3_st = BasicConv3d(self.feats_size[2], embed_channels, 1)
        self.translayer4_st = BasicConv3d(self.feats_size[3], embed_channels, 1)

        self.attention3 = SRMAttention(embed_channels, self.channel_first, 16, **kwargs)
        self.attention2 = SRMAttention(embed_channels, self.channel_first, 8,  **kwargs)
        self.attention1 = SRMAttention(embed_channels, self.channel_first, 4, **kwargs)

        self.res = lambda x, size: F.interpolate(x, size=size, mode='trilinear', align_corners=False)

        self.out_conv1 = UnetOutBlock(spatial_dims, embed_channels, out_chans)

    def proj_feat(self, x):
        new_view = [x.size(0)] + self.proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()
        return x

    def forward(self, x_in):
        base_size = x_in.shape[-3:]

        features = self.encoder(x_in)
        x1 = features[0]  # 2, 48, 112, 112, 32
        x2 = features[1]  # 2, 96, 56, 56, 56
        x3 = features[2]  # 2, 192, 28, 28, 8
        x4 = features[3]  # 2, 384, 14, 14, 4
        x4_st = self.translayer4_st(x4)
        a4 = self.out_conv1(x4_st)

        x1_st = self.translayer1_st(x1)
        x2_st = self.translayer2_st(x2)
        x3_st = self.translayer3_st(x3)

        a3 = self.attention3(x3_st, a4)
        a2 = self.attention2(x2_st, a3)
        a1 = self.attention1(x1_st, a2)

        out4 = self.res(a4, base_size)
        out3 = self.res(a3, base_size)
        out2 = self.res(a2, base_size)
        out1 = self.res(a1, base_size)
        return out1, out2, out3, out4

