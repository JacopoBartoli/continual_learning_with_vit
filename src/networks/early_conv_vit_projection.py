from torch import nn
import torch.nn.functional as F
from .early_conv_vit_net import EarlyConvViT


class Early_conv_vit_proj(nn.Module):

    def __init__(self, num_classes=100, pretrained=False):
        super().__init__()

        self.model = EarlyConvViT(
            dim=768,
            num_classes=2,
            depth = 12,
            heads = 12,
            mlp_dim = 2048,
            channels=3,
        )

        #import ipdb; ipdb.set_trace()
        if pretrained:
            raise NotImplementedError

        # Add projection head
        self.projection = nn.Sequential(
            nn.Linear(in_features = 768, out_features = 2048),
            nn.ReLU(),
            nn.Linear(in_features =2048, out_features = 128)
        )

        # last classifier layer (head) with as many outputs as classes
        self.fc = nn.Linear(in_features=128, out_features=num_classes, bias=True)
        # and `head_var` with the name of the head, so it can be removed when doing incremental learning experiments
        self.head_var = 'fc'

    def forward(self, x):
        x = self.model(x)
        h = self.fc(self.projection(x))
        return h


def early_conv_vit_proj(num_out=100, pretrained=False):
    if pretrained:
        return Early_conv_vit_proj(num_out, pretrained)
    else:
        raise NotImplementedError
    assert 1==0, "you should not be here :/"

