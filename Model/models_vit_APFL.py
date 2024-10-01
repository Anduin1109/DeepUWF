
from functools import partial
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions

import timm.models.vision_transformer
from .util import lr_decay as lrd



class VisionTransformerAPFL(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=True, **kwargs):
        super(VisionTransformerAPFL, self).__init__(**kwargs)
        # global_pool = True:
        norm_layer = kwargs['norm_layer']
        embed_dim = kwargs['embed_dim']
        self.fc_norm = norm_layer(embed_dim)
        del self.norm  # remove the original norm


    def init_apfl(self, apfl_alpha=0.25, num_blocks=1):
        '''
        num_block: the number of attention blocks to be federally trained
        '''
        # APFL
        self.alpha = apfl_alpha
        self.last_block = self.blocks[-num_blocks:]
        self.blocks_forward = self.blocks[:-num_blocks]
        # combine the last block, head and fc_norm as a single module
        self.classifier = nn.Sequential(self.last_block, self.fc_norm, self.head)
        self.classifier_personal = deepcopy(self.classifier)

    def get_optimizers(self, args):
        '''
        return optimizer, optimizer_personal
        '''
        param_groups = lrd.param_groups_lrd(self.modules, args.weight_decay,
            no_weight_decay_list=self.modules.no_weight_decay(),
            layer_decay=args.layer_decay
        )
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
        optimizer_personal = torch.optim.AdamW(self.classifier_personal.parameters(), lr=args.lr)
        return optimizer, optimizer_personal

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
    
        # attention blocks
        for blk in self.blocks_forward:
            x = blk(x)
        return x
    
    def pred(self, x):
        x = self.classifier[0](x)
        x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
        out = self.classifier[1:](x)    # [B, num_classes]
        return out
    
    def pred_personal(self, x):
        x = self.classifier_personal[0](x)
        x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
        out = self.classifier_personal[1:](x)    # [B, num_classes]
        return out


    def forward(self, x, **kwargs):
        x = self.forward_features(x)  # feature: [B, 1024]
        y, y_personal = self.pred(x), self.pred_personal(x)
        return y, y_personal
    


def vit_large_patch16(**kwargs):
    model = VisionTransformerAPFL(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

