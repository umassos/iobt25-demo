#!/usr/bin/env python
# -*- coding: utf-8 -*-


import math
import logging
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchinfo import summary
from torch.hub import load_state_dict_from_url
from typing import List, Tuple, Union, Dict

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

BlockArgs = collections.namedtuple('BlockArgs', [
    'num_repeat', 'kernel_size', 'stride', 'expand_ratio',
    'input_filters', 'output_filters', 'se_ratio', 'id_skip'])

GlobalParams = collections.namedtuple('GlobalParams', [
    'width_coefficient', 'depth_coefficient', 'image_size', 'dropout_rate',
    'num_classes', 'batch_norm_momentum', 'batch_norm_epsilon',
    'drop_connect_rate', 'depth_divisor', 'min_depth', 'include_top'])

PARAM_DICT = {
    # Coefficients:   width,depth,res,dropout
    'efficientnet-b0': (1.0, 1.0, 224, 0.2),
    'efficientnet-b1': (1.0, 1.1, 240, 0.2),
    'efficientnet-b2': (1.1, 1.2, 260, 0.3),
    'efficientnet-b3': (1.2, 1.4, 300, 0.3),
    'efficientnet-b4': (1.4, 1.8, 380, 0.4),
    'efficientnet-b5': (1.6, 2.2, 456, 0.4),
    'efficientnet-b6': (1.8, 2.6, 528, 0.5),
    'efficientnet-b7': (2.0, 3.1, 600, 0.5),
    'efficientnet-b8': (2.2, 3.6, 672, 0.5),
    'efficientnet-l2': (4.3, 5.3, 800, 0.5),
}

MODEL_URLS = {
    'efficientnet-b0':
        'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth',
    'efficientnet-b1':
        'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b1-f1951068.pth',
    'efficientnet-b2':
        'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b2-8bb594d6.pth',
    'efficientnet-b3':
        'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b3-5fb5a3c3.pth',
    'efficientnet-b4':
        'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b4-6ed6700e.pth',
    'efficientnet-b5':
        'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b5-b6417697.pth',
    'efficientnet-b6':
        'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b6-c76e70fd.pth',
    'efficientnet-b7':
        'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b7-dcc49843.pth',
}

EFFICIENTNET_B0_BLOCK_ARGS = [
    BlockArgs(num_repeat=1, kernel_size=3, stride=1, expand_ratio=1, input_filters=32,
              output_filters=16, se_ratio=.25, id_skip=True),
    BlockArgs(num_repeat=2, kernel_size=3, stride=2, expand_ratio=6, input_filters=16,
              output_filters=24, se_ratio=.25, id_skip=True),
    BlockArgs(num_repeat=2, kernel_size=5, stride=2, expand_ratio=6, input_filters=24,
              output_filters=40, se_ratio=.25, id_skip=True),
    BlockArgs(num_repeat=3, kernel_size=3, stride=2, expand_ratio=6, input_filters=40,
              output_filters=80, se_ratio=.25, id_skip=True),
    BlockArgs(num_repeat=3, kernel_size=5, stride=1, expand_ratio=6, input_filters=80,
              output_filters=112, se_ratio=.25, id_skip=True),
    BlockArgs(num_repeat=4, kernel_size=5, stride=2, expand_ratio=6, input_filters=112,
              output_filters=192, se_ratio=.25, id_skip=True),
    BlockArgs(num_repeat=1, kernel_size=3, stride=1, expand_ratio=6, input_filters=192,
              output_filters=320, se_ratio=.25, id_skip=True)
]


def get_global_param(model_name):
    """ Return GlobalParam for a given model """
    width_coefficient, depth_coefficient, image_size, dropout_rate = PARAM_DICT[model_name]

    global_params = GlobalParams(
        width_coefficient=width_coefficient,
        depth_coefficient=depth_coefficient,
        image_size=224,
        dropout_rate=dropout_rate,
        num_classes=365,
        batch_norm_momentum=0.99,
        batch_norm_epsilon=1e-3,
        drop_connect_rate=.3,
        depth_divisor=8,
        min_depth=None,
        include_top=True,
    )
    return global_params


def round_filters(filters, global_params):
    """Calculate and round number of filters based on width multiplier.
       Use width_coefficient, depth_divisor and min_depth of global_params.
    Args:
        filters (int): Filters number to be calculated.
        global_params (namedtuple): Global params of the model.
    Returns:
        new_filters: New filters number after calculating.
    """
    multiplier = global_params.width_coefficient
    if not multiplier:
        return filters

    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    filters *= multiplier
    min_depth = min_depth or divisor  # pay attention to this line when using min_depth
    # follow the formula transferred from official TensorFlow implementation
    new_filters = max(min_depth, int(
        filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:  # prevent rounding by more than 10%
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats, global_params):
    """Calculate module's repeat number of a block based on depth multiplier.
       Use depth_coefficient of global_params.
    Args:
        repeats (int): num_repeat to be calculated.
        global_params (namedtuple): Global params of the model.
    Returns:
        new repeat: New repeat number after calculating.
    """
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    # follow the formula transferred from official TensorFlow implementation
    return int(math.ceil(multiplier * repeats))


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


def drop_connect(inputs: torch.Tensor, p: float, training: bool):
    """Drop connect """
    if not training:
        return inputs

    batch_size = inputs.shape[0]
    keep_prob = 1 - p

    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1],
                                dtype=inputs.dtype, device=inputs.device)
    binary_tensor = torch.floor(random_tensor)

    output = inputs / keep_prob * binary_tensor
    return output


class AnytimeSubNetwork(nn.Module):
    """ An anytime sub-network consisted of two parts:
            1. Feature extractor
            2. Classifier / Predictor
        This module should return the output of both parts
    """

    def __init__(self,
                 feature_extractor: nn.Module,
                 classifier: nn.Module,
                 include_features=False):
        super(AnytimeSubNetwork, self).__init__()
        self.include_features = include_features
        self.feature_extractor = feature_extractor
        self.classifier = classifier

    def forward(self, inputs: torch.Tensor):
        features = self.feature_extractor(inputs)
        output = self.classifier(features)

        if self.include_features:
            return output, features
        else:
            return output


class MBConvBlock(nn.Module):
    """Mobile Inverted Residual Bottleneck Block.
    Args:
        block_args (namedtuple): BlockArgs, defined in utils.py.
        global_params (namedtuple): GlobalParam, defined in utils.py.
        image_size (tuple or list): [image_height, image_width].
    References:
        [1] https://arxiv.org/abs/1704.04861 (MobileNet v1)
        [2] https://arxiv.org/abs/1801.04381 (MobileNet v2)
        [3] https://arxiv.org/abs/1905.02244 (MobileNet v3)
    """

    def __init__(self, block_args, global_params):
        super().__init__()
        self._block_args = block_args
        # pytorch's difference from tensorflow
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (
            0 < self._block_args.se_ratio <= 1)
        # whether to use skip connection and drop connect
        self.id_skip = block_args.id_skip

        # Expansion phase (Inverted Bottleneck)
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * \
            self._block_args.expand_ratio  # number of output channels
        if self._block_args.expand_ratio != 1:
            self._expand_conv = get_same_padding_conv2d(
                in_channels=inp, out_channels=oup, kernel_size=1)
            self._bn0 = nn.BatchNorm2d(
                num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        self._depthwise_conv = get_same_padding_conv2d(in_channels=oup, out_channels=oup,
                                                       groups=oup, kernel_size=k, stride=s)
        self._bn1 = nn.BatchNorm2d(
            num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            num_squeezed_channels = max(
                1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = get_same_padding_conv2d(in_channels=oup, out_channels=num_squeezed_channels,
                                                      kernel_size=1, bias=True)
            self._se_expand = get_same_padding_conv2d(in_channels=num_squeezed_channels, out_channels=oup,
                                                      kernel_size=1, bias=True)

        # Pointwise convolution phase
        final_oup = self._block_args.output_filters
        self._project_conv = get_same_padding_conv2d(
            in_channels=oup, out_channels=final_oup, kernel_size=1)
        self._bn2 = nn.BatchNorm2d(
            num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish = nn.SiLU(inplace=True)

    def forward(self, inputs, drop_connect_rate=None):
        """MBConvBlock's forward function.
        Args:
            inputs (tensor): Input tensor.
            drop_connect_rate (bool): Drop connect rate (float, between 0 and 1).
        Returns:
            Output of this block after processing.
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._expand_conv(inputs)
            x = self._bn0(x)
            x = self._swish(x)

        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self._swish(x)

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_reduce(x_squeezed)
            x_squeezed = self._swish(x_squeezed)
            x_squeezed = self._se_expand(x_squeezed)
            x = torch.sigmoid(x_squeezed) * x

        # Pointwise Convolution
        x = self._project_conv(x)
        x = self._bn2(x)

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            # The combination of skip connection and drop connect brings about stochastic depth.
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate,
                                 training=self.training)
            x = x + inputs  # skip connection
        return x


class AnytimeEfficientNetDepth(nn.Module):
    """EfficientNet model.
       Most easily loaded with the .from_name or .from_pretrained methods.
    Args:
        blocks_args (list[namedtuple]): A list of BlockArgs to construct blocks.
        global_params (namedtuple): A set of GlobalParams shared between blocks.
    References:
        [1] https://arxiv.org/abs/1905.11946 (EfficientNet)
    """

    def __init__(self, blocks_args=None, global_params=None, start_block: int = 0, end_block: int = -1,
                 num_inputs: int = 1):
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args
        self._start_block = start_block
        self._end_block = end_block if 0 < end_block <= len(
            blocks_args) else len(blocks_args)
        self._num_inputs = num_inputs
        self._stem = None
        self._classifier = None
        self.last_channels = 0

        assert self._start_block < self._end_block

        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Get stem static or dynamic convolution depending on image size
        image_size = global_params.image_size

        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(self._global_params.dropout_rate)
        self._swish = nn.SiLU(inplace=True)

        # Stem
        if self._start_block == 0:
            in_channels = 3  # rgb
            # number of output channels
            out_channels = round_filters(32, self._global_params)
            self._conv_stem = get_same_padding_conv2d(
                in_channels, out_channels, kernel_size=3, stride=2)
            self._bn0 = nn.BatchNorm2d(
                num_features=out_channels, momentum=bn_mom, eps=bn_eps)

            self._stem = nn.Sequential(self._conv_stem, self._bn0, self._swish)

        # Build blocks
        self._blocks = nn.ModuleList([])
        for block_idx in range(self._start_block, self._end_block):
            block_args = self._blocks_args[block_idx]
            block_stride = block_args.stride

            # Update block input and output filters based on depth multiplier.
            if block_idx == self._start_block:
                block_args = block_args._replace(
                    input_filters=round_filters(
                        block_args.input_filters, self._global_params) * self._num_inputs,
                    output_filters=round_filters(
                        block_args.output_filters, self._global_params),
                    num_repeat=round_repeats(
                        block_args.num_repeat, self._global_params)
                )
            else:
                block_args = block_args._replace(
                    input_filters=round_filters(
                        block_args.input_filters, self._global_params),
                    output_filters=round_filters(
                        block_args.output_filters, self._global_params),
                    num_repeat=round_repeats(
                        block_args.num_repeat, self._global_params)
                )

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlock(block_args, self._global_params))

            if block_args.num_repeat > 1:  # modify block_args to keep same output size
                block_args = block_args._replace(
                    input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(
                    block_args, self._global_params))

            # Put an exit for each resolution shrink
            # if block_idx in self._exit_blocks:
            #     exit_index = str(len(self._blocks) - 1)
            #     self._output_layers[exit_index] = nn.Sequential(
            #         self._avg_pooling,
            #         nn.Flatten(start_dim=1),
            #         self._dropout,
            #         nn.Linear(block_args.output_filters, self._global_params.num_classes)
            #     )
            #     self._output_layer_index.append(exit_index)
            #     self._sub_networks.append(nn.Sequential(*sub_network))
            #     sub_network = []
            self.last_channels = block_args.output_filters

        # Head
        if self._end_block == len(self._blocks_args):
            in_channels = block_args.output_filters  # output of final block
            out_channels = round_filters(1280, self._global_params)
            self.last_channels = 1280
            self._conv_head = get_same_padding_conv2d(
                in_channels, out_channels, kernel_size=1)
            self._bn1 = nn.BatchNorm2d(
                num_features=out_channels, momentum=bn_mom, eps=bn_eps)

            self._classifier = nn.Sequential(
                self._conv_head,
                self._bn1,
                self._swish,
                self._avg_pooling,
                nn.Flatten(start_dim=1),
                self._dropout,
                nn.Linear(out_channels, self._global_params.num_classes)
            )

    @property
    def num_subnetworks(self):
        return len(self._output_layers)

    @property
    def num_blocks(self):
        return len(self._blocks)

    def forward(self, inputs):
        """EfficientNet's forward function.
           Calls extract_features to extract features, applies final linear layer, and returns logits.
        Args:
            inputs (tensor): Input tensor.
        Returns:
            Output of this model after processing.
        """
        # Stem
        if self._stem:
            x = self._stem(inputs)
        else:
            x = inputs

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                # scale drop connect_rate
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)

        # Head
        if self._classifier:
            x = self._classifier(x)

        return x


class EfficientNetHead(nn.Module):
    def __init__(self, num_classes: int, input_filters: int, drop_out_rate: float = 0.3):
        super(EfficientNetHead, self).__init__()
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


class EfficientNetHeadMulti(nn.Module):
    def __init__(self, num_classes: int, input_filters: int, drop_out_rate: float = 0.3):
        super(EfficientNetHeadMulti, self).__init__()
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


class LinearClassifier(nn.Module):
    def __init__(self, num_classes: int, input_filters: int, drop_out_rate: float = 0.3):
        super(LinearClassifier, self).__init__()
        self._num_classes = num_classes
        self._input_filters = input_filters
        self._drop_out_rate = drop_out_rate

        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(self._drop_out_rate)
        self._linear = nn.Linear(input_filters, self._num_classes)

        self._classifier = nn.Sequential(
            self._avg_pooling,
            nn.Flatten(start_dim=1),
            self._dropout,
            self._linear
        )

    def forward(self, inputs):
        return self._classifier(inputs)


class EnsembleEfficientNet(nn.Module):
    def __init__(self, num_classes: int, cut_point: int, width_ratio: float = 1.0, num_repeat: int = None,
                 last_channels: int = 1280):
        super(EnsembleEfficientNet, self).__init__()
        self.num_classes = num_classes
        self.num_repeat = num_repeat  # Number of repeat for the final block
        self.last_channels = last_channels
        self._cut_point = cut_point
        self.width_ratio = width_ratio

        # Individual upstream models
        self.encoder1 = EfficientNetEarly(
            num_classes=num_classes, cut_point=cut_point, last_channels=last_channels)
        self.encoder2 = EfficientNetEarly(
            num_classes=num_classes, cut_point=cut_point, last_channels=last_channels)
        self.encoder3 = EfficientNetEarly(
            num_classes=num_classes, cut_point=cut_point, last_channels=last_channels)

        # 2-combination classifiers
        self.classifier12 = EfficientNetHead(
            self.num_classes, self.encoder1.encoder.last_channels + self.encoder2.encoder.last_channels)
        self.classifier13 = EfficientNetHead(
            self.num_classes, self.encoder1.encoder.last_channels + self.encoder3.encoder.last_channels)
        self.classifier23 = EfficientNetHead(
            self.num_classes, self.encoder2.encoder.last_channels + self.encoder3.encoder.last_channels)

        # 3-combination classifier
        total_input_filters = self.encoder1.encoder.last_channels + \
            self.encoder2.encoder.last_channels + self.encoder3.encoder.last_channels
        self.classifier_comb = EfficientNetHeadMulti(
            self.num_classes, total_input_filters)

        # self.encoder1_stat = summary(self.encoder1.encoder, verbose=0)
        # self.encoder2_stat = summary(self.encoder2.encoder, verbose=0)
        # self.classifier1_stat = summary(self.encoder1.classifier, verbose=0)
        # self.classifier2_stat = summary(self.encoder2.classifier, verbose=0)
        # self.classifier_comb_stat = summary(self.classifier_comb, verbose=0)

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

        print("Ensemble EfficientNet B0 created")
        print(
            f"Encoder-1 # params: {self._encoder1_params}, last channels {self.encoder1.last_channels}")
        print(
            f"Encoder-2 # params: {self._encoder2_params}, last channels {self.encoder2.last_channels}")
        print(
            f"Encoder-3 # params: {self._encoder3_params}, last channels {self.encoder3.last_channels}")
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


class EfficientNetEarly(nn.Module):
    def __init__(self, num_classes: int, cut_point: int, use_head: bool = False, last_channels: int = 1280):
        super(EfficientNetEarly, self).__init__()
        self.num_classes = num_classes
        self._cut_point = cut_point
        self.use_head = use_head
        self.last_channels = last_channels

        self.encoder = get_multiexit_efficientnet_b0(
            self.num_classes, 0, self._cut_point, 1, 1)

        if use_head:
            self.classifier = EfficientNetHead(
                self.num_classes, self.encoder.last_channels, self.last_channels)
        else:
            self.classifier = LinearClassifier(
                self.num_classes, self.encoder.last_channels)

    def forward(self, inputs):
        self._x = self.encoder(inputs)
        logits = self.classifier(self._x)

        return logits


def get_efficientnet_b0_block_args(width_ratio: float = 1.0):
    block_args = EFFICIENTNET_B0_BLOCK_ARGS.copy()
    for i in range(len(block_args)):
        if i != 0:
            block_args[i] = block_args[i]._replace(input_filters=int(block_args[i].input_filters * width_ratio),
                                                   output_filters=int(block_args[i].output_filters * width_ratio))
        else:
            block_args[i] = block_args[i]._replace(
                output_filters=int(block_args[i].output_filters * width_ratio))

    return block_args


def get_multiexit_efficientnet_b0(num_classes: int, start_block: int, end_block: int,
                                  num_inputs: int, width_ratio: float = 1.0) -> AnytimeEfficientNetDepth:
    """ Get anytime efficientnet-b0 model """
    global_params = get_global_param("efficientnet-b0")
    global_params = global_params._replace(num_classes=num_classes)
    block_args = get_efficientnet_b0_block_args(width_ratio)
    model = AnytimeEfficientNetDepth(block_args, global_params, start_block=start_block,
                                     end_block=end_block, num_inputs=num_inputs)

    return model


def get_efficientnet_encoder12(num_classes: int, start_block: int, end_block: int,
                               num_inputs: int, width_ratio: float = 1.0,
                               num_repeat: int = None) -> AnytimeEfficientNetDepth:
    """ Get anytime efficientnet-b0 model """
    global_params = get_global_param("efficientnet-b0")
    global_params = global_params._replace(num_classes=num_classes)
    block_args = get_efficientnet_b0_block_args(width_ratio)

    if num_repeat:
        block_args[end_block - 1] = block_args[end_block -
                                               1]._replace(num_repeat=num_repeat)

    model = AnytimeEfficientNetDepth(block_args, global_params, start_block=start_block,
                                     end_block=end_block, num_inputs=num_inputs)

    return model
