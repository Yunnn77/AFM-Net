# models/attention_blocks.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class BasicConvBlock(nn.Module):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        activation: bool = True,
        use_bn: bool = True,
    ):
        super(BasicConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=not use_bn,
        )
        self.bn = nn.BatchNorm2d(out_planes) if use_bn else nn.Identity()
        self.activation = nn.ReLU(inplace=True) if activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.bn(self.conv(x)))

class DepthwiseSeparableConvBlock(nn.Module):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        activation: bool = True,
        use_bn: bool = True,
    ):
        super(DepthwiseSeparableConvBlock, self).__init__()
        # Depthwise convolution
        self.depthwise = nn.Conv2d(
            in_planes,
            in_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_planes,
            bias=False,
        )
        self.bn_dw = nn.BatchNorm2d(in_planes) if use_bn else nn.Identity()
        self.relu_dw = nn.ReLU(inplace=True) if activation else nn.Identity()

        # Pointwise convolution
        self.pointwise = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn_pw = nn.BatchNorm2d(out_planes) if use_bn else nn.Identity()
        self.relu_pw = nn.ReLU(inplace=True) if activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.bn_dw(x)
        x = self.relu_dw(x)
        x = self.pointwise(x)
        x = self.bn_pw(x)
        x = self.relu_pw(x)
        return x

class ChannelAttentionCBAM(nn.Module):
    def __init__(self, in_planes: int, ratio: int = 16):
        super(ChannelAttentionCBAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, kernel_size=1, bias=False),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttentionEnhanced(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super(SpatialAttentionEnhanced, self).__init__()
        self.conv_block = nn.Sequential(
            BasicConvBlock(2, 4, kernel_size=3, padding=1, activation=True),
            nn.Conv2d(
                in_channels=4,
                out_channels=1,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False,
            ),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv_block(x_cat)
        return self.sigmoid(out)

class BottleneckBlock(nn.Module):
    """Standard Bottleneck block (used when fedab_use_dwsc=False)."""

    expansion: int = 4

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        dilation: int = 1,
    ):
        super(BottleneckBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class BottleneckWithDWSCBlock(nn.Module):
    """Bottleneck block using Depthwise Separable Convolution."""

    expansion: int = 4

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        dilation: int = 1,
    ):
        super(BottleneckWithDWSCBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2_dwsc = DepthwiseSeparableConvBlock(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            activation=True,
            use_bn=True,
        )

        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.conv2_dwsc(out)  # DWSC includes activation internally
        out = self.bn3(self.conv3(out))  # No activation here, wait for residual sum

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class FEDAB_Enhanced_DWSC(nn.Module):
    def __init__(
        self,
        in_channels: int,
        bottleneck_planes: int,
        out_channels_after_att: int,
        num_bottleneck_blocks: int = 1,
        reduction_ratio_ca: int = 16,
        spatial_kernel_sa: int = 7,
        use_dwsc_for_branches: bool = True,
    ):
        super(FEDAB_Enhanced_DWSC, self).__init__()

        self.use_dwsc_for_branches = use_dwsc_for_branches
        CurrentBottleneckBlock = (
            BottleneckWithDWSCBlock if use_dwsc_for_branches else BottleneckBlock
        )

        initial_1x1_out_channels = (
            bottleneck_planes * CurrentBottleneckBlock.expansion
        )
        self.conv_1x1_input = BasicConvBlock(
            in_channels,
            initial_1x1_out_channels,
            kernel_size=1,
            padding=0,
        )

        self.branch1 = self._make_layer(
            CurrentBottleneckBlock,
            initial_1x1_out_channels,
            bottleneck_planes,
            num_bottleneck_blocks,
            stride=1,
            dilation=1,
        )
        self.branch2 = self._make_layer(
            CurrentBottleneckBlock,
            initial_1x1_out_channels,
            bottleneck_planes,
            num_bottleneck_blocks,
            stride=1,
            dilation=2,
        )

        branch3_out_channels = bottleneck_planes * CurrentBottleneckBlock.expansion
        if use_dwsc_for_branches:
            self.branch3 = DepthwiseSeparableConvBlock(
                initial_1x1_out_channels,
                branch3_out_channels,
                kernel_size=3,
                padding=1,
            )
        else:
            self.branch3 = BasicConvBlock(
                initial_1x1_out_channels,
                branch3_out_channels,
                kernel_size=3,
                padding=1,
            )

        branch_out_channels = bottleneck_planes * CurrentBottleneckBlock.expansion
        concatenated_channels = branch_out_channels * 3

        if use_dwsc_for_branches:
            self.fusion_conv = DepthwiseSeparableConvBlock(
                concatenated_channels,
                concatenated_channels,
                kernel_size=1,
                padding=0,
            )
        else:
            self.fusion_conv = BasicConvBlock(
                concatenated_channels,
                concatenated_channels,
                kernel_size=1,
                padding=0,
            )

        self.ca = ChannelAttentionCBAM(
            concatenated_channels, ratio=reduction_ratio_ca
        )
        self.sa = SpatialAttentionEnhanced(kernel_size=spatial_kernel_sa)

        self.conv_1x1_output = BasicConvBlock(
            concatenated_channels,
            out_channels_after_att,
            kernel_size=1,
            padding=0,
            activation=False,
        )

        if in_channels != out_channels_after_att:
            self.shortcut_conv = BasicConvBlock(
                in_channels,
                out_channels_after_att,
                kernel_size=1,
                padding=0,
                activation=False,
            )
        else:
            self.shortcut_conv = nn.Identity()

        self.final_relu = nn.ReLU(inplace=True)

    def _make_layer(
        self,
        block_type,
        in_planes: int,
        planes: int,
        blocks: int,
        stride: int = 1,
        dilation: int = 1,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or in_planes != planes * block_type.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    planes * block_type.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block_type.expansion),
            )

        layers = [
            block_type(
                in_planes,
                planes,
                stride,
                downsample,
                dilation=dilation,
            )
        ]
        in_planes_current = planes * block_type.expansion
        for _ in range(1, blocks):
            layers.append(
                block_type(
                    in_planes_current,
                    planes,
                    dilation=dilation,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        shortcut = self.shortcut_conv(x_in)

        x = self.conv_1x1_input(x_in)
        b1_out = self.branch1(x)
        b2_out = self.branch2(x)
        b3_out = self.branch3(x)

        x_concat = torch.cat((b1_out, b2_out, b3_out), dim=1)
        x_fused = self.fusion_conv(x_concat)

        x_ca = self.ca(x_fused) * x_fused
        x_sa = self.sa(x_ca) * x_ca

        x_out = self.conv_1x1_output(x_sa)
        return self.final_relu(x_out + shortcut)