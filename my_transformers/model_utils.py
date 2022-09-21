# re-written modules for use
import torch
from torch import nn

from einops import rearrange, repeat


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
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
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., orginal_head=False):
        super().__init__()
        self.original_head = orginal_head

        if not orginal_head:
            inner_dim = dim_head *  heads
            project_out = not (heads == 1 and dim_head == dim)

            self.heads = heads
            self.scale = dim_head ** -0.5

            self.attend = nn.Softmax(dim = -1)
            self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

            self.to_out = nn.Sequential(
                nn.Linear(inner_dim, dim),
                nn.Dropout(dropout)
            ) if project_out else nn.Identity()

        else:
            ...



    def forward(self, x):
        # print("Att ", x.shape)  # [1, 65, 1024]
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        # print(qkv[0].shape) # [1, 65, 1024]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)  # h d stands for dim_head and head
        # print(q.shape)  # [1, 16, 65, 64]

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # print(dots.shape)  # [1, 16, 65, 65]

        attn = self.attend(dots)
        # print(attn.shape)  # [1, 16, 65, 65]
        weights = attn

        out = torch.matmul(attn, v)
        # print(out.shape)  # [1, 16, 65, 64]
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        # print(out.shape)  # [1, 65, 1024]
        return out, weights

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., offset=True, ff=True):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.offset = offset
        self.ff = ff
        print("Transformer model offset status {}".format(offset))
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        attn_weights = []
        for attn, ff in self.layers:
            attn_x, weights = attn(x)
            attn_weights.append(weights)
            if self.offset:
                x = attn_x # + x
            else:
                x = attn_x + x
            if self.ff:
                x = ff(x) + x
        return x, attn_weights


def pixel_upsample(x, H, W, size=2):
    B, N, C = x.size()
    assert N == H*W
    x = x.permute(0, 2, 1)
    x = x.view(-1, C, H, W)
    x = nn.PixelShuffle(size)(x)
    B, C, H, W = x.size()
    x = x.view(-1, C, H*W)
    x = x.permute(0,2,1)
    return x, H, W

def bicubic_upsample(x, H, W, size=2):
    # example x shape: (B, 64, 384), the H and W here are size 8
    B, N, C = x.size()
    assert N == H*W
    x = x.permute(0, 2, 1)
    x = x.view(-1, C, H, W)
    x = nn.functional.interpolate(x, scale_factor=size, mode='bicubic')
    B, C, H, W = x.size()
    x = x.view(-1, C, H*W)
    x = x.permute(0,2,1)
    return x, H, W

def img_shuffule(img, perm):
    # img shape is [B, 3, 256, 256] for example
    img_c, img_w, img_h = img.shape[1], img.shape[-1], img.shape[-2]

    img_flat = rearrange(img, 'b c h d -> (b c) (h d)', c=img_c)
    # perm = torch.randperm(img_w*img_h)
    b_range = torch.arange(img.shape[0] * img.shape[1], device=img.device)[:, None]
    img_flat = img_flat[b_range, perm]
    img_shuf = rearrange(img_flat, '(b c) (h d) -> b c h d', h=img_h, c=img_c)
    # print("check", torch.equal(img, img_shuf))
    return img_shuf

def img_unshuffle(img, perm):
    img_c, img_w, img_h = img.shape[1], img.shape[-1], img.shape[-2]

    img_flat = rearrange(img, 'b c h d -> (b c) (h d)')
    recover_img = torch.zeros(img_flat.shape, device=img.device)
    b_range = torch.arange(img.shape[0]*img.shape[1], device=img.device)[:, None]

    # print(recover_img.shape, img_flat.shape)
    recover_img[b_range, perm] = img_flat
    img_unshuf = rearrange(recover_img, '(b c) (h d) -> b c h d', h=img_h, c=img_c)
    return img_unshuf

def img_grid(img, grid=16, patch=16):
    """256*256 img --> 16*16 * 256 blocks"""
    # img shape b c h w
    img_c, img_w, img_h = img.shape[1], img.shape[2], img.shape[3]
    assert grid*patch == img_h and img_h == img_w

    # first perm rows (within rows) by grids
    # then perm cols by grids
    rows_perm = [i + grid * j for i in range(grid) for j in range(patch)]
    cols_perm = [i + grid * j for i in range(grid) for j in range(patch)]

    img = img[:, :, :, rows_perm]
    img = img[:, :, cols_perm, :]
    return img


def img_ungrid(img, grid=16, patch=16):
    img_b, img_c, img_w, img_h = img.shape[0], img.shape[1], img.shape[2], img.shape[3]

    rows_perm = [i + grid * j for i in range(grid) for j in range(patch)]
    cols_perm = [i + grid * j for i in range(grid) for j in range(patch)]

    img_un1 = torch.zeros(img.shape, device=img.device)
    img_un2 = torch.zeros(img.shape, device=img.device)

    img_un1[:, :, :, rows_perm] = img
    img_un2[:, :, cols_perm, :] = img_un1

    return img_un2