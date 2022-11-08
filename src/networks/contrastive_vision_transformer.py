import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

__all__ = ['contrastive_vision_transformer']

class Shrink(torch.nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride
        self.pool = nn.AvgPool1d(1,stride,0)

    def forward(self, x):
        #B, N, C = x.shape

        x = x.transpose(1,2)
        x = self.pool(x)
        x = x.transpose(1,2)

        return x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
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

    def __init__(self, dim, num_heads=8, dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5

        self.to_qv = nn.Linear(dim, dim * 2, bias=False)
        self.attn_normalization = nn.BatchNorm2d(num_heads)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Sequential(
            nn.Linear(dim,dim),
            nn.Dropout(dropout)

        )

        # Implementation of External Attention mechanism.
        self.ext_k = None
        self.ext_bias = None

    def forward(self, x):
        B, N, C = x.shape

        qv = self.to_qv(x).chunk(2, dim = -1)
        q,v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.num_heads), qv)

        if self.ext_k is None:
            b, h, n, d = v.shape
            self.ext_k = nn.Parameter(torch.randn(1, h, n, d)).to(v.device)
        if self.ext_bias is None:
            b, h, n, d = v.shape
            self.ext_bias = nn.Parameter(torch.randn(1, h, n, n)).to(v.device)


        attn = (q @ self.ext_k.transpose(-2, -1) + self.ext_bias) * self.scale
        
        attn = self.attn_normalization(attn)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v)

        x = rearrange(x, 'b h n d -> b n (h d)')

        x = self.proj(x)

        return x

class Transformer(nn.Module):

    def __init__(self, dim, depth, num_heads, mlp_dim, dropout = 0.):
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


class CVTBlock(nn.Module):
    def __init__(self,
                 dim=768,
                 depth=2,
                 num_heads=8,
                 mlp_dim=2048,
                 dropout=0.,
                 ):
        
        super().__init__()
        self.blocks = nn.ModuleList([
            Transformer(dim=dim, depth = depth, num_heads=num_heads, mlp_dim=mlp_dim, dropout=dropout)
            for i in range(depth)])

    def forward(self,x):
        for blk in self.blocks:
            x = blk(x)

        return x
        
class EmbeddingsHook(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.identity = nn.Identity()
        def forward(self, *args):
            x = self.identity(*args)            
            return x
            
class ContrastiveVisionTransformer(nn.Module):    

    def __init__(self,
                 embedding_dim=768,
                 n_input_channels=3,
                 n_conv_layers=1,
                 kernel_size=7,
                 stride=2,
                 padding=3,
                 pooling_kernel_size=3,
                 pooling_stride=2,
                 pooling_padding=1,
                 dropout=0.,
                 depth=2,
                 num_heads=8,
                 mlp_dim=2048,
                 projection_dim=128,
                 num_classes=100,
                 *args, **kwargs):
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

        self.embedding_dim = embedding_dim
        self.resolution = stride * stride
        self.sequence_length = None
        self.seq_pool = True

        self.n_input_channels = n_input_channels

        self.attention_pool = nn.Linear(self.embedding_dim, 1)
        self.embeddings_hook = EmbeddingsHook()


        self.positional_emb = None

        self.dropout = nn.Dropout(p=dropout)

        self.cvt1 = CVTBlock(dim=embedding_dim, depth = depth, num_heads=num_heads, mlp_dim=mlp_dim, dropout=dropout)
        self.shrink1 = Shrink(stride=stride)
        self.cvt2 = CVTBlock(dim=embedding_dim, depth = depth, num_heads=num_heads, mlp_dim=mlp_dim, dropout=dropout)
        self.shrink2 = Shrink(stride=stride)
        self.cvt3 = CVTBlock(dim=embedding_dim, depth = depth, num_heads=num_heads, mlp_dim=mlp_dim, dropout=dropout)

        self.norm = nn.LayerNorm(embedding_dim)


        # Learnable focuses.
        self.learnable_focuses = nn.Parameter(torch.randn(num_classes, self.embedding_dim))

        

        self.projection_head = nn.Sequential(
            nn.Linear(embedding_dim, mlp_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, projection_dim)
        )

        # `head_var` with the name of the head, so it can be removed when doing incremental learning experiments
        self.fc = nn.Linear(projection_dim, num_classes)

        self.head_var = 'fc'


        self.apply(self.init_weight)

    def forward(self, x):
        x = self.tokenizer(x)

        if self.sequence_length is None:
            self.sequence_length = self.tokenizer.sequence_length
        if self.positional_emb is None and x.size(1) < self.sequence_length:
            x = F.pad(x, (0, 0, 0, self.n_input_channels - x.size(1)), mode='constant', value=0)

        x = self.dropout(x)

        x = self.cvt1(x)
        x = self.shrink1(x)

        x = self.cvt2(x)
        x = self.shrink2(x)

        x = self.cvt3(x)

        x = self.norm(x)

        
        x = torch.matmul(F.softmax(self.attention_pool(x), dim=1).transpose(-1, -2), x).squeeze(-2)

        z = self.embeddings_hook(x)

        x = self.projection_head(z)

        # Pass the data through the classification head.
        h = self.fc(x)
        return h

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

def contrastive_vision_transformer(pretrained=False, **kwargs):
    if pretrained:
        raise NotImplementedError
    model = ContrastiveVisionTransformer(num_classes=100,
            embedding_dim=768,
            num_heads = 8,
            mlp_dim = 2048,
            channels=3, 
            dropout=0.1,
            stride=2,**kwargs)
    return model