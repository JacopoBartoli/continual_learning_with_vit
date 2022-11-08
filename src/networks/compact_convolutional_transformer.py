import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

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

    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Sequential(
            nn.Linear(dim,dim),
            nn.Dropout(dropout)

        )

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q,k,v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.num_heads), qkv)


        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v)

        x = rearrange(x, 'b h n d -> b n (h d)')

        x = self.proj(x)

        return x

class Transformer(nn.Module):

    def __init__(self, dim, depth, num_heads, mlp_dim, dropout = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, num_heads=num_heads, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class Tokenizer(nn.Module):
    def __init__(self,
                 kernel_size, stride, padding,
                 pooling_kernel_size=3, pooling_stride=2, pooling_padding=1,
                 n_conv_layers=1,
                 n_input_channels=3,
                 n_output_channels=64,
                 in_planes=64,
                 activation=None,
                 max_pool=True,
                 conv_bias=False):
        super().__init__()

        n_filter_list = [n_input_channels] + \
                        [in_planes for _ in range(n_conv_layers - 1)] + \
                        [n_output_channels]

        self.conv_layers = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(n_filter_list[i], n_filter_list[i + 1],
                          kernel_size=(kernel_size, kernel_size),
                          stride=(stride, stride),
                          padding=(padding, padding), bias=conv_bias),
                nn.Identity() if activation is None else activation(),
                nn.MaxPool2d(kernel_size=pooling_kernel_size,
                             stride=pooling_stride,
                             padding=pooling_padding) if max_pool else nn.Identity()
            )
                for i in range(n_conv_layers)
            ])

        self.flattener = nn.Flatten(2, 3)

        self.apply(self.init_weight)
    
    def sequence_length(self, n_channels=3, height=32, width=32):
        return self.forward(torch.zeros((1, n_channels, height, width))).shape[1]


    def forward(self, x):
        B, C, H, W = x.shape
        self.sequence_length = self.flattener(self.conv_layers(torch.zeros((1, C, H, W)).to(x.device))).transpose(-2,-1).shape[1]
        x = self.conv_layers(x)
        x = self.flattener(x).transpose(-2,-1)
        return x

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)


class CCT(nn.Module):
    def __init__(self,
                img_size=32,
                 embedding_dim=768,
                 n_input_channels=3,
                 n_conv_layers=2,
                 kernel_size=3,
                 stride=2,
                 padding=3,
                 pooling_kernel_size=3,
                 pooling_stride=2,
                 pooling_padding=1,
                 dropout=0.1,
                 depth=7,
                 num_heads=8,
                 mlp_dim=2048,
                 num_classes=100,
                 positional_embeddings = 'learnable',
                 *args, **kwargs
                 ):
        
        super().__init__()
        self.tokenizer = Tokenizer(n_input_channels=n_input_channels,
                                   n_output_channels=embedding_dim,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   pooling_kernel_size=pooling_kernel_size,
                                   pooling_stride=pooling_stride,
                                   pooling_padding=pooling_padding,
                                   max_pool=True,
                                   activation=nn.ReLU,
                                   n_conv_layers=n_conv_layers,
                                   conv_bias=False)

        self.dropout = nn.Dropout(p=dropout)

        self.attention_pool = nn.Linear(embedding_dim, 1)

        sequence_length=self.tokenizer.sequence_length(n_channels=n_input_channels,
                                                           height=img_size,
                                                           width=img_size)

        if positional_embeddings == 'learnable':
            self.positional_emb = nn.Parameter(torch.zeros(1, sequence_length, embedding_dim),
                                                requires_grad=True)
            nn.init.trunc_normal_(self.positional_emb, std=0.2)

        self.blocks = nn.ModuleList([
            Transformer(dim=embedding_dim, depth = depth, num_heads=num_heads, mlp_dim=mlp_dim, dropout=dropout)
            for i in range(depth)])

        
        self.norm = nn.LayerNorm(embedding_dim)

        self.fc = nn.Linear(embedding_dim, num_classes)

        self.head_var = 'fc'

        

    def forward(self,x):
        x = self.tokenizer(x)

        if self.positional_emb is not None:
            x += self.positional_emb

        x = self.dropout(x)
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        x = torch.matmul(F.softmax(self.attention_pool(x), dim=1).transpose(-1, -2), x).squeeze(-2)

        x = self.fc(x)

        return x

def compact_convolutional_transformer(pretrained=False, **kwargs):
    if pretrained:
        raise NotImplementedError
    model = CCT(num_classes=100,
            embedding_dim=768,
            num_heads = 4,
            mlp_dim = 2048,
            channels=3, 
            dropout=0.1,
            kernel_size=3,
            n_conv_layers=2,**kwargs)
    return model