import torch
from PIL import Image
import numpy as np

import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import efficientnet

from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

import sys

sys.path.insert(1, "3rdparty/pytorch-image-models")
from ensemble_efficient_net_b0 import (
    EnsembleEfficientNet,
    get_multiexit_efficientnet_b0,
)
from ensemble_resnet_50 import EnsembleResnet50

from ensemble_vit import EnsembleViT
from ensemble_deepspeech2 import EnsembleDeepSpeech2

# Load the pretrained ResNet-50 model
blocks = 5
architecture_name = f"EENetB0" ### 7 Blocks
# architecture_name = f"ERNet50"  ### 4 Blocks
# architecture_name = f"EViT" ### 12 (6) Blocks
#architecture_name = f"EDeepSp"  ### 6 Blocks

num_classes = 608 

head_name = f"FC"
model_dir = f"models"
model_name = f"ensemble-effnet-c5-lr-0.005-tin"
model_file = f"model_best.pth.tar"
initial_checkpoint = f"{model_dir}/{model_name}/{model_file}"


# model_name = f"{architecture_name}_{blocks}_{head_name}"
if architecture_name == "EENetB0":
    model = EnsembleEfficientNet(num_classes=num_classes, cut_point=blocks)
elif architecture_name == "ERNet50":
    model = EnsembleResnet50(num_classes=num_classes, cut_point=blocks)
elif architecture_name == "EViT":
    model = EnsembleViT(num_classes=num_classes, cut_point=blocks, image_size=224)
elif architecture_name == "EDeepSp":
    model = EnsembleDeepSpeech2(
        n_feats=128,
        n_tokens=29,
        num_rnn_layers=blocks,
        hidden_size=512,
        rnn_dropout=0.1,
    )
else:
    raise ValueError(f"Unknown architecture name: {architecture_name}")

if initial_checkpoint:
    checkpoint = torch.load(initial_checkpoint, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])

model.eval()  # Set the model to evaluation mode

# Generate random input data
# ResNet-50 expects input size of (3, 224, 224)
input_tensor = torch.randn(1, 3, 224, 224)
if architecture_name == "EDeepSp":
    input_tensor = torch.randn(1, 1, 128, 31)

encoder_tensors = {
    "EENetB0": {
        1: torch.randn([1, 16, 112, 112]),# 200704
        2: torch.randn([1,24,56,56]), # 75264
        3: torch.randn([1, 40, 28, 28]),
        4: torch.randn([1, 80, 14, 14]),
        5: torch.randn([1, 112, 14, 14]),
        6: torch.randn([1, 192, 7, 7]),
        7: torch.randn([1, 320, 7, 7]),
    },
    "ERNet50": {
        1: torch.randn([1, 256, 56, 56]),
        2: torch.randn([1, 512, 28, 28]),
        3: torch.randn([1, 1024, 14, 14]),
        4: torch.randn([1, 2048, 7, 7]),
        5: torch.randn([1, 2048, 7, 7]),
    },
    "EViT": {
        1: torch.randn([1, 768]),
        2: torch.randn([1, 768]),
        3: torch.randn([1, 768]),
        4: torch.randn([1, 768]),
        5: torch.randn([1, 768]),
        6: torch.randn([1, 768]),
        12: torch.randn([1, 768]),
    },
    "EDeepSp": {
        1: torch.randn([8, 1, 512]),
        2: torch.randn([8, 1, 512]),
        3: torch.randn([8, 1, 512]),
        4: torch.randn([8, 1, 512]),
        5: torch.randn([8, 1, 512]),
        6: torch.randn([8, 1, 512]),
    },
}

# Run inference
with torch.no_grad():
    y_comb = model(input_tensor)

# Print the output
# print("Output shape:", y_comb.shape)

# torch.onnx.export(
#     model,
#     (input_tensor,),
#     f"models/{model_name}/single.onnx",
#     input_names=["input"],
#     output_names=["output"],
# )
# torch.onnx.export(
#     model.encoder1.encoder,
#     (input_tensor,),
#     f"models/{model_name}/encoder1.onnx",
#     input_names=["input"],
#     output_names=["enc1_output"],
# )
# torch.onnx.export(
#     model.encoder1.classifier,
#     (encoder_tensors[architecture_name][blocks],),
#     f"models/{model_name}/classifier1.onnx",
#     input_names=["enc1_output"],
#     output_names=["cl1_output"],
# )
# torch.onnx.export(
#     model.encoder2.encoder,
#     (input_tensor,),
#     f"models/{model_name}/encoder2.onnx",
#     input_names=["input"],
#     output_names=["enc2_output"],
# )
# torch.onnx.export(
#     model.encoder2.classifier,
#     (encoder_tensors[architecture_name][blocks],),
#     f"models/{model_name}/classifier2.onnx",
#     input_names=["enc2_output"],
#     output_names=["cl2_output"],
# )
# torch.onnx.export(
#     model.classifier_comb,
#     (
#         encoder_tensors[architecture_name][blocks],
#         encoder_tensors[architecture_name][blocks],
#     ),
#     f"models/{model_name}/head.onnx",
#     input_names=["enc1_output", "enc2_output"],
#     output_names=["head_output"],
# )

# Get pretrained model
model2 = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

# If you need different number of classes, you can modify the final layer
# model.classifier = torch.nn.Linear(model.classifier.in_features, 608)

model2.eval()  # Set the model to evaluation mode

# # model2 = efficientnet.efficientnet_b0(num_classes=1000)
# # model2.eval()  # Set the model to evaluation mode
with torch.no_grad():
    y_comb = model2(input_tensor)

torch.onnx.export(model2, (input_tensor,), f"{model_dir}/original.onnx",input_names=["input"],output_names=["output"])

# model_name = "SplitEfficientNet-B0"

# any_time_c1 = get_multiexit_efficientnet_b0(num_classes=100, start_block=0, end_block=5, num_inputs=1)
# any_time_c1.eval()  # Set the model to evaluation mode

# with torch.no_grad():
#     y = any_time_c1(input_tensor)

# torch.onnx.export(any_time_c1, (input_tensor,), f"system/{model_name}_1-5.onnx",input_names=["input"],output_names=["output"])


# input_tensor_c2 = torch.randn(1,112,14,14)


# any_time_c23 = get_multiexit_efficientnet_b0(num_classes=100, start_block=5, end_block=6, num_inputs=1)
# any_time_c23.eval()  # Set the model to evaluation mode

# with torch.no_grad():
#     y = any_time_c23(input_tensor_c2)

# torch.onnx.export(any_time_c23, (input_tensor_c2,), f"system/{model_name}_C6.onnx",input_names=["input"],output_names=["output"])


# input_tensor_c4 = torch.randn(1,192,7,7)


# any_time_c47 = get_multiexit_efficientnet_b0(num_classes=100, start_block=6, end_block=7, num_inputs=1)
# any_time_c47.eval()  # Set the model to evaluation mode

# with torch.no_grad():
#     y = any_time_c47(input_tensor_c4)

# torch.onnx.export(any_time_c47, (input_tensor_c4,), f"system/{model_name}_7-C.onnx",input_names=["input"],output_names=["output"])
