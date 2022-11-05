from torch import nn
import torch.nn.functional as F
import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

__all__ = ['econv_vit_focus']

# classes
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=7, dim_head=64, dropout=0.1):
        """
        reduced the default number of heads by 1 per https://arxiv.org/pdf/2106.14881v2.pdf
        """
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        # Implementation of External Attention mechanism.
        self.ext_k = None
        self.ext_bias = None


    def forward(self, x):
     
        qv = self.to_qv(x).chunk(2, dim = -1)
        q, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qv)

        if self.ext_k is None:
            b, h, n, d = v.shape
            self.ext_k = nn.Parameter(torch.randn(1, h, n, d)).to(v.device)
        if self.ext_bias is None:
            b, h, n, d = v.shape
            self.ext_bias = nn.Parameter(torch.randn(1, h, n, n)).to(v.device)

        dots = (torch.matmul(q, self.ext_k.transpose(-1, -2)) + self.ext_bias) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class EmbeddingsHook(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.identity = nn.Identity()
    def forward(self, *args):
        x = self.identity(*args)            
        return x

class Econv_vit_focus(nn.Module):
    
    def __init__(self, *, num_classes, dim, depth, heads, mlp_dim, projection_dim, pool='cls', channels=3, dim_head=64, dropout=0.1, emb_dropout=0.1):
        """
        3x3 conv, stride 1, 5 conv layers per https://arxiv.org/pdf/2106.14881v2.pdf

        Learnable focuses for avoid forgetting of past embeddings.
        """
        super().__init__()

        n_filter_list = (channels, 24, 48, 96, 196)  # hardcoding for now because that's what the paper used

        self.conv_layers = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(in_channels=n_filter_list[i],
                          out_channels=n_filter_list[i + 1],
                          kernel_size=3,  # hardcoding for now because that's what the paper used
                          stride=2,  # hardcoding for now because that's what the paper used
                          padding=1),  # hardcoding for now because that's what the paper used
            )
                for i in range(len(n_filter_list)-1)
            ])

        self.conv_layers.add_module("conv_1x1", torch.nn.Conv2d(in_channels=n_filter_list[-1], 
                                    out_channels=dim, 
                                    stride=1,  # hardcoding for now because that's what the paper used 
                                    kernel_size=1,  # hardcoding for now because that's what the paper used 
                                    padding=0))  # hardcoding for now because that's what the paper used
        self.conv_layers.add_module("flatten image", 
                                    Rearrange('batch channels height width -> batch (height width) channels'))
        self.pos_embedding = nn.Parameter(torch.randn(1, n_filter_list[-1] + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.learnable_focuses = nn.Parameter(torch.randn(num_classes, dim))

        self.projection_head = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, projection_dim)
        )

        self.pool = pool
        self.embeddings_hook = EmbeddingsHook()

        self.fc = nn.Linear(in_features=projection_dim, out_features=num_classes, bias=True)
        # and `head_var` with the name of the head, so it can be removed when doing incremental learning experiments
        self.head_var = 'fc'


    def forward(self, img):
        x = self.conv_layers(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.embeddings_hook(x)

        x = self.projection_head(x)

        x = self.fc(x)

        return x

def econv_vit_focus(pretrained=False, **kwargs):
    if pretrained:
        raise NotImplementedError
    model = Econv_vit_focus(dim=768,
            num_classes=100,
            depth=12,
            heads=8,
            mlp_dim = 2048,
            channels=3,
            projection_dim=128,**kwargs)
    return model
