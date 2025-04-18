import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.utils import DropPath, to_2tuple, trunc_normal_

from torch import einsum
from einops import rearrange, reduce, repeat


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        if out_features==None:
            out_features = in_features
        if hidden_features==None:
            hidden_features = in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qkv_scale=None, attn_drop=0., proj_drop=0., with_qkv=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qkv_scale or head_dim ** -0.5
        self.with_qkv = with_qkv
        if self.with_qkv:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        B, N, C = x.shape
        if self.with_qkv:
           qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
           q, k, v = qkv[0], qkv[1], qkv[2]
        else:
           qkv = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
           q, k, v  = qkv, qkv, qkv

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        if self.with_qkv:
           x = self.proj(x)
           x = self.proj_drop(x)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qkv_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm):

        super().__init__()
        self.attention_type = "divided_space_time"

        # Spatial Attention
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qkv_scale=qkv_scale, attn_drop=attn_drop)

        # Temporal Attention
        self.temporal_norm1 = norm_layer(dim)
        self.temporal_attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qkv_scale=qkv_scale, attn_drop=attn_drop, proj_drop=drop)
        self.temporal_fc = nn.Linear(dim, dim)

        ## drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, B, T, W):
        num_spatial_tokens = (x.size(1) - 1) // T
        H = num_spatial_tokens // W

        # Temporal
        xt = x[:, 1:, :]
        xt = rearrange(xt, 'b (h w t) m -> (b h w) t m',b=B,h=H,w=W,t=T)
        res_temporal = self.drop_path(self.temporal_attn(self.temporal_norm1(xt)))
        res_temporal = rearrange(res_temporal, '(b h w) t m -> b (h w t) m',b=B,h=H,w=W,t=T)
        res_temporal = self.temporal_fc(res_temporal)
        xt = x[:, :1, :] + res_temporal

        # Spatial
        init_cls_token = x[:,0,:].unsqueeze(1)
        cls_token = init_cls_token.repeat(1, T, 1)
        cls_token = rearrange(cls_token, 'b t m -> (b t) m',b=B,t=T).unsqueeze(1)
        xs = xt
        xs = rearrange(xs, 'b (h w t) m -> (b t) (h w) m',b=B,h=H,w=W,t=T)
        xs = torch.cat((cls_token, xs), 1)
        res_spatial = self.drop_path(self.attn(self.norm1(xs)))

        ### Taking care of CLS token
        cls_token = res_spatial[:,0,:]
        cls_token = rearrange(cls_token, '(b t) m -> b t m',b=B,t=T)
        cls_token = torch.mean(cls_token,1,True) ## averaging for every frame
        res_spatial = res_spatial[:,1:,:]
        res_spatial = rearrange(res_spatial, '(b t) (h w) m -> b (h w t) m',b=B,h=H,w=W,t=T)
        res = res_spatial
        x = xt

        ## Mlp
        x = torch.cat((init_cls_token, x), 1) + torch.cat((cls_token, res), 1)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PatchEmbed(nn.Module):
    def __init__(self, img_size=(224, 224), patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        # img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.proj(x)
        W = x.size(-1)
        x = x.flatten(2).transpose(1, 2)
        return x, T, W

class VisionTransformer(nn.Module):
    def __init__(self, img_size=(224, 224), patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qkv_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, hybrid_backbone=None, norm_layer=nn.LayerNorm, num_frames=8, dropout=0.):

        super().__init__()
        self.depth = depth
        self.dropout = nn.Dropout(dropout)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        ## Positional Embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.time_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
        self.time_drop = nn.Dropout(p=drop_rate)


        ## Attention Blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qkv_scale=qkv_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(self.depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.pos_embed = trunc_normal_(self.pos_embed, std=.02)
        self.cls_token = trunc_normal_(self.cls_token, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        '''
            Input:
            x: shape (B, C, T, H, W)
            B: batch size
            C: num of channels
            T: number of frames
            H: height of image
            W: width of image
        '''
        B = x.shape[0]
        x, T, W = self.patch_embed(x) # x ((b t) x patch_num x embed_dim), T: num_frames, W:patch_num
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)

        x = torch.cat((cls_tokens, x), dim=1) # x.shape[-1] is number of patches + 1

        ## resizing the positional embeddings in case they don't match the input at inference
        if x.size(1) != self.pos_embed.size(1):
            pos_embed = self.pos_embed
            cls_pos_embed = pos_embed[0,0,:].unsqueeze(0).unsqueeze(1)
            other_pos_embed = pos_embed[0,1:,:].unsqueeze(0).transpose(1, 2)
            P = int(other_pos_embed.size(2) ** 0.5)
            H = x.size(1) // W
            other_pos_embed = other_pos_embed.reshape(1, x.size(2), P, P)
            new_pos_embed = F.interpolate(other_pos_embed, size=(H, W), mode='nearest')
            new_pos_embed = new_pos_embed.flatten(2)
            new_pos_embed = new_pos_embed.transpose(1, 2)
            new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)
            x = x + new_pos_embed
        else:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        ## Time Embeddings
        cls_tokens = x[:B, 0, :].unsqueeze(1)
        x = x[:,1:]
        x = rearrange(x, '(b t) n m -> (b n) t m',b=B,t=T)
        ## Resizing time embeddings in case they don't match
        if T != self.time_embed.size(1):
            time_embed = self.time_embed.transpose(1, 2)
            new_time_embed = F.interpolate(time_embed, size=(T), mode='nearest')
            new_time_embed = new_time_embed.transpose(1, 2)
            x = x + new_time_embed
        else:
            x = x + self.time_embed
        x = self.time_drop(x)
        x = rearrange(x, '(b n) t m -> b (n t) m',b=B,t=T)
        x = torch.cat((cls_tokens, x), dim=1)

        ## Attention blocks
        for blk in self.blocks:
            x = blk(x, B, T, W)

        x = self.norm(x)
        return x[:, 0]
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x



