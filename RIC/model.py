import torch
import sys
sys.path.append('..')

from Model import models_vit

def load_model_from_checkpoint(checkpoint_path, num_classes, img_size=224, device='cuda', model_name='vit_large_patch16'):
    model = models_vit.__dict__[model_name](
        img_size=img_size,
        num_classes=num_classes,
        drop_path_rate=0.1,
        global_pool=True,
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu')['model'])
    model = model.to(device)
    return model