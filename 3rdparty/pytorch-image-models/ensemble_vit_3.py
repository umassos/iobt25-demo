# Phil Wang's implementation
# Reference - https://github.com/lucidrains/vit-pytorch/
import torch
from torch import nn

from torchinfo import summary
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads,
                          dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * \
            (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {
            'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                      p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


class ViTEncoder(nn.Module):
    def __init__(self, *, image_size, patch_size=16, num_classes=100, dim=768, depth=12, heads=12, mlp_dim=3072, pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        self._dim = dim

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * \
            (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {
            'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                      p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return x


class ViTEarly(nn.Module):
    def __init__(self, image_size, num_classes: int, cut_point: int):
        super(ViTEarly, self).__init__()
        self._num_classes = num_classes
        self._cut_point = cut_point
        self.encoder = ViTEncoder(
            image_size=image_size, num_classes=num_classes, depth=cut_point)
        self.classifier = nn.Linear(self.encoder._dim, num_classes)

    def forward(self, input):
        self._x = self.encoder(input)
        return self.classifier(self._x)


class ViTHead(nn.Module):
    def __init__(self, num_classes: int, input_filters: int):
        super(ViTHead, self).__init__()
        self._num_classes = num_classes
        self._input_filters = input_filters

        self._mlp_head = nn.Linear(input_filters, num_classes)

    def forward(self, inputs1, inputs2):
        x_comb = torch.cat([inputs1, inputs2], dim=1)
        return self._mlp_head(x_comb)


class ViTHeadMulti(nn.Module):
    def __init__(self, num_classes: int, input_filters: int):
        super(ViTHeadMulti, self).__init__()
        self._num_classes = num_classes
        self._input_filters = input_filters

        self._mlp_head = nn.Linear(input_filters, num_classes)

    def forward(self, inputs1, inputs2, inputs3):
        x_comb = torch.cat([inputs1, inputs2, inputs3], dim=1)
        return self._mlp_head(x_comb)


class EnsembleViT(nn.Module):
    def __init__(self, image_size, num_classes: int, cut_point: int):
        super(EnsembleViT, self).__init__()
        self._num_classes = num_classes
        self._cut_point = cut_point

        self.encoder1 = ViTEarly(image_size, num_classes, cut_point)
        self.encoder2 = ViTEarly(image_size, num_classes, cut_point)
        self.encoder3 = ViTEarly(image_size, num_classes, cut_point)

        self.classifier12 = ViTHead(
            num_classes, self.encoder1.encoder._dim + self.encoder2.encoder._dim)
        self.classifier13 = ViTHead(
            num_classes, self.encoder1.encoder._dim + self.encoder3.encoder._dim)
        self.classifier23 = ViTHead(
            num_classes, self.encoder2.encoder._dim + self.encoder3.encoder._dim)

        self.classifier_comb = ViTHeadMulti(
            num_classes, self.encoder1.encoder._dim + self.encoder2.encoder._dim + self.encoder3.encoder._dim)

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
            f"Encoder-1 # params: {self._encoder1_params}")
        print(
            f"Encoder-2 # params: {self._encoder2_params}")
        print(
            f"Encoder-3 # params: {self._encoder3_params}")

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
