# MIT License
#
# Copyright (c) 2021 Soohwan Kim.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
from torch import Tensor

import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torchinfo import summary

from typing import Tuple


class NormGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate=0.1):
        """
        Args:
            input_dim (int): Input feature dimension.
            hidden_dim (int): Hidden state dimension.
            dropout_rate (float): Dropout rate.
        """
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            dropout=dropout_rate,
            batch_first=False,
            bidirectional=True,
        )
        self.batch_norm = nn.BatchNorm1d(hidden_dim)

    def forward(self, inputs, hs=None):
        """
        Forward pass through GRU and norm layer
        """
        x, hs = self.gru(inputs, hs)
        x = x.view(x.shape[0], x.shape[1], 2, -1).sum(2)
        batch_size, time_steps = x.shape[1], x.shape[0]
        x = x.view(batch_size * time_steps, -1)
        x = self.batch_norm(x)
        x = x.view(time_steps, batch_size, -1).contiguous()

        return x, hs


class DeepSpeech2(nn.Module):
    def __init__(self, n_feats, n_tokens, num_rnn_layers, hidden_size, rnn_dropout):
        """
        Args:
            n_feats (int)
            n_tokens (int)
            num_rnn_layers (int)
            hidden_size (int)
            rnn_dropout (float)
        """
        super().__init__()

        # TODO; automate creation from config
        self.conv_params = {
            "conv1": {"padding": (20, 5), "kernel_size": (41, 11), "stride": (2, 2)},
            "conv2": {"padding": (10, 5), "kernel_size": (21, 11), "stride": (2, 2)},
            "conv3": {"padding": (10, 5), "kernel_size": (21, 11), "stride": (2, 1)},
        }

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, **self.conv_params["conv1"]),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, **self.conv_params["conv2"]),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 96, **self.conv_params["conv3"]),
            nn.BatchNorm2d(96),
            nn.ReLU(),
        )

        rnn_input_dim = self.calc_rnn_input_size(n_feats)
        rnn_input_dim *= 96

        self.rnns = nn.Sequential(
            *[
                NormGRU(
                    (hidden_size if i > 0 else rnn_input_dim), hidden_size, rnn_dropout
                )
                for i in range(num_rnn_layers)
            ]
        )

        self.fc = nn.Linear(hidden_size, n_tokens)

    def calc_rnn_input_size(self, n_feats):
        """
        Calculates the size of the RNN input after convolutions for NormGRU
        Args:
            n_feats (int): Number of input features.
        Returns:
            int: Size of RNN input.
        """
        size = n_feats
        for conv_param in self.conv_params.values():
            size = (
                size + 2 * conv_param["padding"][0] -
                conv_param["kernel_size"][0]
            ) // conv_param["stride"][0] + 1
        return size

    def forward(self, spectrogram, spectrogram_length, **batch):
        """
        Model forward method.

        Args:
            spectrogram (Tensor): input spectrogram.
            spectrogram_length (Tensor): spectrogram original lengths.
        Returns:
            output (dict): output dict containing log_probs and
                transformed lengths.
        """
        x = self.conv(spectrogram.unsqueeze(1))
        x = x.view(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])
        x = (
            x.transpose(1, 2).transpose(0, 1).contiguous()
        )  # n_tokens x batch_size x (n_channels * n_freqs)

        h = None
        for rnn in self.rnns:
            x, h = rnn(x, h)

        time_steps, batch_size = x.shape[0], x.shape[1]
        x = x.view(time_steps * batch_size, -1)
        logits = self.fc(x)
        logits = logits.view(time_steps, batch_size, -1).transpose(0, 1)

        log_probs = nn.functional.log_softmax(logits, dim=-1)
        return {
            "logits": logits,
            "log_probs": log_probs,
            "log_probs_length": self.transform_input_lengths(spectrogram_length),
        }

    def transform_input_lengths(self, input_lengths):
        """
        As the network may compress the Time dimension, we need to know
        what are the new temporal lengths after compression.

        Args:
            input_lengths (Tensor): old input lengths
        Returns:
            transformed input_lengths (Tensor): new temporal lengths
        """
        for conv_param in self.conv_params.values():
            input_lengths = (
                input_lengths
                + 2 * conv_param["padding"][1]
                - conv_param["kernel_size"][1]
            ) // conv_param["stride"][1] + 1
        return input_lengths

    def __str__(self):
        """
        Return model details including parameter counts.
        """
        all_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel()
                               for p in self.parameters() if p.requires_grad)

        result_str = super().__str__()
        result_str += f"\nAll parameters: {all_params}"
        result_str += f"\nTrainable parameters: {trainable_params}"
        return result_str


class DeepSpeech2Encoder(nn.Module):
    def __init__(self, n_feats, n_tokens, num_rnn_layers, hidden_size, rnn_dropout):
        """
        Args:
            n_feats (int)
            n_tokens (int)
            num_rnn_layers (int)
            hidden_size (int)
            rnn_dropout (float)
        """
        super().__init__()

        # TODO; automate creation from config
        self.conv_params = {
            "conv1": {"padding": (20, 5), "kernel_size": (41, 11), "stride": (2, 2)},
            "conv2": {"padding": (10, 5), "kernel_size": (21, 11), "stride": (2, 2)},
            "conv3": {"padding": (10, 5), "kernel_size": (21, 11), "stride": (2, 1)},
        }

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, **self.conv_params["conv1"]),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, **self.conv_params["conv2"]),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 96, **self.conv_params["conv3"]),
            nn.BatchNorm2d(96),
            nn.ReLU(),
        )

        rnn_input_dim = self.calc_rnn_input_size(n_feats)
        rnn_input_dim *= 96

        self.rnns = nn.Sequential(
            *[
                NormGRU(
                    (hidden_size if i > 0 else rnn_input_dim), hidden_size, rnn_dropout
                )
                for i in range(num_rnn_layers)
            ]
        )

        # self.fc = nn.Linear(hidden_size, n_tokens)

    def calc_rnn_input_size(self, n_feats):
        """
        Calculates the size of the RNN input after convolutions for NormGRU
        Args:
            n_feats (int): Number of input features.
        Returns:
            int: Size of RNN input.
        """
        size = n_feats
        for conv_param in self.conv_params.values():
            size = (
                size + 2 * conv_param["padding"][0] -
                conv_param["kernel_size"][0]
            ) // conv_param["stride"][0] + 1
        return size

    def transform_input_lengths(self, input_lengths):
        """
        As the network may compress the Time dimension, we need to know
        what are the new temporal lengths after compression.

        Args:
            input_lengths (Tensor): old input lengths
        Returns:
            transformed input_lengths (Tensor): new temporal lengths
        """
        for conv_param in self.conv_params.values():
            input_lengths = (
                input_lengths
                + 2 * conv_param["padding"][1]
                - conv_param["kernel_size"][1]
            ) // conv_param["stride"][1] + 1
        return input_lengths

    def forward(self, spectrogram):
        """
        Model forward method.

        Args:
            spectrogram (Tensor): input spectrogram.
            spectrogram_length (Tensor): spectrogram original lengths.
        Returns:
            output (dict): output dict containing log_probs and
                transformed lengths.
        """
        x = self.conv(spectrogram)
        x = x.view(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])
        x = (
            x.transpose(1, 2).transpose(0, 1).contiguous()
        )  # n_tokens x batch_size x (n_channels * n_freqs)

        h = None
        for rnn in self.rnns:
            x, h = rnn(x, h)

        self._x = x
        return x

    def __str__(self):
        """
        Return model details including parameter counts.
        """
        all_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel()
                               for p in self.parameters() if p.requires_grad)

        result_str = super().__str__()
        result_str += f"\nAll parameters: {all_params}"
        result_str += f"\nTrainable parameters: {trainable_params}"
        return result_str


class DeepSpeech2Early(nn.Module):
    def __init__(self, n_feats, n_tokens, num_rnn_layers, hidden_size, rnn_dropout):
        """
        Args:
            n_feats (int)
            n_tokens (int)
            num_rnn_layers (int)
            hidden_size (int)
            rnn_dropout (float)
        """
        super().__init__()

        self.encoder = DeepSpeech2Encoder(
            n_feats, n_tokens, num_rnn_layers, hidden_size, rnn_dropout)

        self.classifier = nn.Linear(hidden_size, n_tokens)

    def calc_rnn_input_size(self, n_feats):
        """
        Calculates the size of the RNN input after convolutions for NormGRU
        Args:
            n_feats (int): Number of input features.
        Returns:
            int: Size of RNN input.
        """
        size = n_feats
        for conv_param in self.conv_params.values():
            size = (
                size + 2 * conv_param["padding"][0] -
                conv_param["kernel_size"][0]
            ) // conv_param["stride"][0] + 1
        return size

    def forward(self, spectrogram):
        """
        Model forward method.

        Args:
            spectrogram (Tensor): input spectrogram.
            spectrogram_length (Tensor): spectrogram original lengths.
        Returns:
            output (dict): output dict containing log_probs and
                transformed lengths.
        """

        input_lengths = torch.tensor([m.shape[-1] for m in spectrogram])
        self._x = x = self.encoder(spectrogram)
        time_steps, batch_size = x.shape[0], x.shape[1]
        x = x.view(time_steps * batch_size, -1)
        logits = self.classifier(x)
        logits = logits.view(time_steps, batch_size, -1)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs, self.encoder.transform_input_lengths(input_lengths)
        # return {
        #     "logits": logits,
        #     "log_probs": log_probs,
        #     "log_probs_length": self.transform_input_lengths(spectrogram_length),
        # }

    def __str__(self):
        """
        Return model details including parameter counts.
        """
        all_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel()
                               for p in self.parameters() if p.requires_grad)

        result_str = super().__str__()
        result_str += f"\nAll parameters: {all_params}"
        result_str += f"\nTrainable parameters: {trainable_params}"
        return result_str


class DeepSpeech2Head(nn.Module):
    def __init__(
        self,
        input_filters: int,
        num_classes: int,
        device: torch.device = 'cuda',
    ):
        super(DeepSpeech2Head, self).__init__()
        self.device = device
        self.fc = nn.Linear(input_filters, num_classes)

    def forward(self, inputs1, inputs2):

        x_comb = torch.cat([inputs1, inputs2], dim=-1)
        time_steps, batch_size = x_comb.shape[0], x_comb.shape[1]
        x_comb = x_comb.view(time_steps * batch_size, -1)
        logits = self.fc(x_comb)
        logits = logits.view(time_steps, batch_size, -1)
        log_probs = F.log_softmax(logits, dim=-1)

        return log_probs


class EnsembleDeepSpeech2(nn.Module):
    def __init__(self, n_feats, n_tokens, num_rnn_layers, hidden_size, rnn_dropout):
        super(EnsembleDeepSpeech2, self).__init__()

        self.encoder1 = DeepSpeech2Early(
            n_feats, n_tokens, num_rnn_layers, hidden_size, rnn_dropout)

        self.encoder2 = DeepSpeech2Early(
            n_feats, n_tokens, num_rnn_layers, hidden_size, rnn_dropout)

        self.classifier_comb = DeepSpeech2Head(hidden_size*2, n_tokens)

        self.encoder1_stat = summary(self.encoder1.encoder, verbose=0)
        self.encoder2_stat = summary(self.encoder2.encoder, verbose=0)
        self.classifier1_stat = summary(self.encoder1.classifier, verbose=0)
        self.classifier2_stat = summary(self.encoder2.classifier, verbose=0)
        self.classifier_comb_stat = summary(self.classifier_comb, verbose=0)

        print("Ensemble Deepspeech created")
        print(
            f"Encoder-1 # params: {self.encoder1_stat.total_params}, last channels {hidden_size}")
        print(
            f"Encoder-2 # params: {self.encoder2_stat.total_params}, last channels {hidden_size}")
        print(f"Classifier-1 # params: {self.classifier1_stat.total_params}")
        print(f"Classifier-2 # params: {self.classifier2_stat.total_params}")
        print(
            f"Classifier-comb # params: {self.classifier_comb_stat.total_params}")

    def forward(self, inputs):
        # Individual branch outputs
        y1, output_lengths = self.encoder1(inputs)
        y2, _ = self.encoder2(inputs)
        self._output_lengths = output_lengths

        # Intermediate representations
        y_comb = self.classifier_comb(self.encoder1._x, self.encoder2._x)

        return y1, y2, y_comb

    def get_output_lengths(self):
        return self._output_lengths

    def initialize_encoders(self, nn1_checkpoint_path: str = '', nn2_checkpoint_path: str = '', device: str = 'cpu'):
        # Load the checkpoints from the given path
        # and update encoder state dicts
        if nn1_checkpoint_path:
            checkpt_nn1 = torch.load(nn1_checkpoint_path, map_location=device)
            self.encoder1.load_state_dict(checkpt_nn1['state_dict'])

        if nn2_checkpoint_path:
            checkpt_nn2 = torch.load(nn2_checkpoint_path, map_location=device)
            self.encoder2.load_state_dict(checkpt_nn2['state_dict'])

    def freeze_and_unfreeze_encoders(self, freeze_nn1: bool = False, freeze_nn2: bool = False):
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
