#!/usr/bin/env python3
"""
    Created date: 6/5/24
"""

import torch
import torch.nn as nn

from torchinfo import summary
from typing import List, Tuple, Union, Callable, Type


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """ 1x1 convolutional layer """
    return nn.Conv2d(in_planes, out_planes, 1, stride=stride, bias=False)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, padding=1) -> nn.Conv2d:
    """ 3x3 convolutional layer """
    return nn.Conv2d(in_planes, out_planes, 3, stride=stride, padding=padding, bias=False)


def get_same_padding_conv2d(in_channels: int,
                            out_channels: int,
                            kernel_size: int = 1,
                            groups: int = 1,
                            stride: int = 1,
                            p: int = None,
                            bias: bool = False) -> nn.Conv2d:
    """ Same padding Conv2d """
    if not p:
        p = (kernel_size - 1) // 2
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                     stride=stride, groups=groups, bias=bias, padding=p)


class Bottleneck(nn.Module):
    # Resnet bottleneck architecture
    expansion = 4

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int = 1,
                 downsample: bool = False,
                 norm_layer: nn.Module = nn.BatchNorm2d
                 ) -> None:
        super(Bottleneck, self).__init__()

        self.conv1 = conv1x1(inplanes, planes)
        self.conv2 = conv3x3(planes, planes, stride=stride)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.downsample = None

        self.inplanes = inplanes
        self.planes = planes
        self.bn1 = norm_layer(self.planes)
        self.bn2 = norm_layer(self.planes)
        self.bn3 = norm_layer(self.expansion * self.planes)

        if downsample:
            self.downsample = nn.Sequential(conv1x1(inplanes, planes * self.expansion, stride=stride),
                                            norm_layer(self.expansion * self.planes))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)
        return out


class LinearClassifier(nn.Module):
    def __init__(self, num_classes: int, input_filters: int):
        super(LinearClassifier, self).__init__()
        self._num_classes = num_classes
        self._input_filters = input_filters

        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._linear = nn.Linear(input_filters, self._num_classes)

        self._classifier = nn.Sequential(
            self._avg_pooling,
            nn.Flatten(start_dim=1),
            self._linear
        )

    def forward(self, inputs):
        return self._classifier(inputs)


# class ResNetHead(nn.Module):
#     def __init__(self, num_classes: int, input_filters: int, out_channels: int):
#         super(ResNetHead, self).__init__()
#         self._num_classes = num_classes
#         self._input_filters = input_filters
#         self._out_channels = out_channels

#         self._conv_head = nn.Conv2d(self._input_filters, self._out_channels,
#                                     kernel_size=1, bias=False, stride=1, padding=0)
#         self._bn1 = nn.BatchNorm2d(self._out_channels)
#         self._relu = nn.ReLU(inplace=True)
#         self._classifier = LinearClassifier(self._num_classes, self._out_channels)

#     def forward(self, inputs):
#         outputs = self._conv_head(inputs)
#         outputs = self._bn1(outputs)
#         outputs = self._relu(outputs)
#         outputs = self._classifier(outputs)

#         return outputs

class ResnetHead(nn.Module):
    def __init__(self, num_classes: int, input_filters: int, drop_out_rate: float = 0.3):
        super(ResnetHead, self).__init__()
        self._num_classes = num_classes
        self._input_filters = input_filters
        self._drop_out_rate = drop_out_rate

        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(self._drop_out_rate)
        self._linear = nn.Linear(self._input_filters, self._num_classes)

        self._classifier_in = nn.Sequential(
            self._avg_pooling,
            nn.Flatten(start_dim=1)
        )
        self._classifier_out = nn.Sequential(
            self._dropout,
            self._linear
        )

    def forward(self, inputs1, inputs2):
        x1 = self._classifier_in(inputs1)
        x2 = self._classifier_in(inputs2)

        x_comb = torch.cat([x1, x2], dim=1)
        return self._classifier_out(x_comb)


class ResnetHeadMulti(nn.Module):
    def __init__(self, num_classes: int, input_filters: int, drop_out_rate: float = 0.3):
        super(ResnetHeadMulti, self).__init__()
        self._num_classes = num_classes
        self._input_filters = input_filters
        self._drop_out_rate = drop_out_rate

        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(self._drop_out_rate)
        self._linear = nn.Linear(self._input_filters, self._num_classes)

        self._classifier_in = nn.Sequential(
            self._avg_pooling,
            nn.Flatten(start_dim=1)
        )
        self._classifier_out = nn.Sequential(
            self._dropout,
            self._linear
        )

    def forward(self, inputs1, inputs2, inputs3):
        x1 = self._classifier_in(inputs1)
        x2 = self._classifier_in(inputs2)
        x3 = self._classifier_in(inputs3)

        x_comb = torch.cat([x1, x2, x3], dim=1)
        return self._classifier_out(x_comb)


class AnytimeResnet50(nn.Module):
    # Anytime implementation of Resnet
    def __init__(self,
                 num_classes: int = 100,
                 start_block: int = 0,
                 end_block: int = -1,
                 num_inputs: int = 1,
                 base_width: int = 64,
                 first_stride: int = 2,
                 use_head: bool = False,
                 head_channels: int = 320,
                 features_only: bool = False,
                 norm_layer: nn.Module = nn.BatchNorm2d) -> None:
        super(AnytimeResnet50, self).__init__()

        self.layer_blocks = [3, 4, 6, 3]
        self.num_classes = num_classes
        self.start_block = start_block
        self.end_block = end_block if 0 < end_block <= len(
            self.layer_blocks) else len(self.layer_blocks)
        self.num_inputs = num_inputs
        self.base_width = base_width
        self.first_stride = first_stride
        self.norm_layer = norm_layer
        self.layer_inplanes = [64, 256, 512, 1024]
        self.layer_planes = [64, 128, 256, 512]
        self.layer_strides = [1, 2, 2, 2]
        self.layers = nn.ModuleList()
        self.stem = None
        self.classifier = None
        self.features_only = features_only
        self.use_head = use_head
        self.head_channels = head_channels
        self.head_channels = self.layer_planes[self.end_block -
                                               1] * Bottleneck.expansion

        assert self.start_block < self.end_block

        # Stem
        if self.start_block == 0:
            self.conv1 = nn.Conv2d(3, self.base_width, kernel_size=(7, 7), stride=(first_stride, first_stride),
                                   padding=(3, 3), bias=False)
            self.bn1 = norm_layer(self.base_width)
            self.relu = nn.ReLU(inplace=True)
            self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            self.stem = nn.Sequential(
                self.conv1, self.bn1, self.relu, self.max_pool)

        # Building blocks
        for i in range(self.start_block, self.end_block):
            inplanes = self.layer_inplanes[i] * \
                self.num_inputs if i == self.start_block and i != 0 else self.layer_inplanes[
                    i]

            self.layers.append(self._make_layer(inplanes,
                                                self.layer_planes[i],
                                                self.layer_blocks[i],
                                                stride=self.layer_strides[i]))

        # Classifier
        if not features_only:
            if use_head:
                self.classifier = ResNetHead(
                    self.num_classes, self.head_channels, self.head_channels)
            else:
                self.classifier = LinearClassifier(
                    num_classes=self.num_classes, input_filters=self.head_channels)

    def _make_layer(self, inplanes: int, planes: int, num_blocks: int, stride: int = 1) -> nn.Sequential:
        """ Make Resnet layer """
        downsample = False

        if stride != 1 or inplanes != planes * Bottleneck.expansion:
            downsample = True

        layers = list()
        cur_block = Bottleneck(inplanes, planes, stride,
                               downsample, self.norm_layer)
        layers.append(cur_block)
        inplanes = planes * Bottleneck.expansion

        for _ in range(1, num_blocks):
            cur_block = Bottleneck(inplanes, planes)
            layers.append(cur_block)

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple:
        """ Inference with width and depth """

        if self.stem:
            x = self.stem(x)

        for layer in self.layers:
            x = layer(x)

        if self.classifier:
            outputs = self.classifier(x)
        else:
            outputs = x

        return outputs


class EnsembleResnet50(nn.Module):
    def __init__(self, num_classes: int, cut_point: int, head_channels: int = 320):
        super(EnsembleResnet50, self).__init__()
        self.num_classes = num_classes
        self._cut_point = cut_point
        self._head_channels = head_channels

        # self.encoder1 = AnytimeResnet50(self.num_classes, 0, self._cut_point, 1, features_only=True)
        # self.encoder2 = AnytimeResnet50(self.num_classes, 0, self._cut_point, 1, features_only=True)
        # self.classifier1 = LinearClassifier(self.num_classes, self.encoder1.head_channels)
        # self.classifier2 = LinearClassifier(self.num_classes, self.encoder2.head_channels)
        self.encoder1 = Resnet50Early(
            self.num_classes, self._cut_point, use_head=False, head_channels=self._head_channels)
        self.encoder2 = Resnet50Early(
            self.num_classes, self._cut_point, use_head=False, head_channels=self._head_channels)
        self.encoder3 = Resnet50Early(
            self.num_classes, self._cut_point, use_head=False, head_channels=self._head_channels)

        self.classifier12 = ResnetHead(self.num_classes, self.encoder1.encoder.head_channels
                                       + self.encoder2.encoder.head_channels)
        self.classifier13 = ResnetHead(self.num_classes, self.encoder1.encoder.head_channels
                                       + self.encoder3.encoder.head_channels)
        self.classifier23 = ResnetHead(self.num_classes, self.encoder2.encoder.head_channels
                                       + self.encoder3.encoder.head_channels)
        # self.encoder12 = AnytimeResnet50(self.num_classes, self._cut_point, self._cut_point+1, 2, features_only=True)
        self.classifier_comb = ResnetHeadMulti(self.num_classes, self.encoder1.encoder.head_channels
                                               + self.encoder2.encoder.head_channels + self.encoder3.encoder.head_channels)

        self._encoder1_params = sum(
            [m.numel() for m in self.encoder1.encoder.parameters()])
        self._encoder2_params = sum(
            [m.numel() for m in self.encoder2.encoder.parameters()])
        self._encoder3_params = sum(
            [m.numel() for m in self.encoder3.encoder.parameters()])

        self._classifier1_params = sum(
            [m.numel() for m in self.encoder1.classifier.parameters()])
        self._classifier2_params = sum(
            [m.numel() for m in self.encoder2.classifier.parameters()])
        self._classifier3_params = sum(
            [m.numel() for m in self.encoder3.classifier.parameters()])

        self._classifier12_params = sum(
            [m.numel() for m in self.classifier12.parameters()])
        self._classifier13_params = sum(
            [m.numel() for m in self.classifier13.parameters()])
        self._classifier23_params = sum(
            [m.numel() for m in self.classifier23.parameters()])

        self._classifier_comb_params = sum(
            [m.numel() for m in self.classifier_comb.parameters()])

        print("Ensemble Resnet50 created")
        print(
            f"Encoder-1 # params: {self._encoder1_params}, last channels {self.encoder1.head_channels}")
        print(
            f"Encoder-2 # params: {self._encoder2_params}, last channels {self.encoder2.head_channels}")
        print(
            f"Encoder-3 # params: {self._encoder3_params}, last channels {self.encoder3.head_channels}")

        print(f"Classifier-1 # params: {self._classifier1_params}")
        print(f"Classifier-2 # params: {self._classifier2_params}")
        print(f"Classifier-3 # params: {self._classifier3_params}")

        print(f"Classifier-12 # params: {self._classifier12_params}")
        print(f"Classifier-13 # params: {self._classifier13_params}")
        print(f"Classifier-23 # params: {self._classifier23_params}")

        print(
            f"Classifier-comb # params: {self._classifier_comb_params}")

    def forward(self, inputs):
        # Individual branch outputs
        y1 = self.encoder1(inputs)
        y2 = self.encoder2(inputs)
        y3 = self.encoder3(inputs)

        y12 = self.classifier12(self.encoder1._x, self.encoder2._x)
        y13 = self.classifier13(self.encoder1._x, self.encoder3._x)
        y23 = self.classifier23(self.encoder2._x, self.encoder3._x)

        # Intermediate representations
        y_comb = self.classifier_comb(
            self.encoder1._x, self.encoder2._x, self.encoder3._x)

        return y1, y2, y3, y12, y13, y23, y_comb

    def initialize_encoders(self, nn1_checkpoint_path: str = '', nn2_checkpoint_path: str = '', nn3_checkpoint_path: str = '', device: str = 'cpu'):
        # Load the checkpoints from the given path
        # and update encoder state dicts
        if nn1_checkpoint_path:
            checkpt_nn1 = torch.load(nn1_checkpoint_path, map_location=device)
            self.encoder1.load_state_dict(checkpt_nn1['state_dict'])

        if nn2_checkpoint_path:
            checkpt_nn2 = torch.load(nn2_checkpoint_path, map_location=device)
            self.encoder2.load_state_dict(checkpt_nn2['state_dict'])

        if nn3_checkpoint_path:
            checkpt_nn3 = torch.load(nn3_checkpoint_path, map_location=device)
            self.encoder3.load_state_dict(checkpt_nn3['state_dict'])

    def freeze_and_unfreeze_encoders(self, freeze_nn1: bool = False, freeze_nn2: bool = False, freeze_nn3: bool = False):
        # Freeze/Unfreeze NN-1 and NN-2 encoder weights
        if freeze_nn1:
            print("Freezing NN-1")
            for param in self.encoder1.parameters():
                param.requires_grad = False
        else:
            print("Not Freezing NN-1")
            for param in self.encoder1.parameters():
                param.requires_grad = True

        if freeze_nn2:
            print("Freezing NN-2")
            for param in self.encoder2.parameters():
                param.requires_grad = False
        else:
            print("Not Freezing NN-2")
            for param in self.encoder2.parameters():
                param.requires_grad = False

        if freeze_nn3:
            print("Freezing NN-3")
            for param in self.encoder3.parameters():
                param.requires_grad = False
        else:
            print("Not Freezing NN-3")
            for param in self.encoder3.parameters():
                param.requires_grad = True


class Resnet50Early(nn.Module):
    def __init__(self, num_classes: int, cut_point: int, use_head: bool = False, head_channels: int = 1280):
        super(Resnet50Early, self).__init__()
        self.num_classes = num_classes
        self._cut_point = cut_point
        self.use_head = use_head
        self.head_channels = head_channels
        use_head = False

        self.encoder = AnytimeResnet50(
            self.num_classes, 0, self._cut_point, 1, features_only=True)

        if use_head:
            self.classifier = ResnetHead(
                self.num_classes, self.encoder.head_channels, self.head_channels)
        else:
            self.classifier = LinearClassifier(
                self.num_classes, self.encoder.head_channels)

    def forward(self, inputs):
        self._x = self.encoder(inputs)
        logits = self.classifier(self._x)

        return logits
