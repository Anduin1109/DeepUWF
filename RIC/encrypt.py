import os
import pathlib
import numpy as np
import torch
from torch.autograd import Variable
import torch.optim as optim
from torchvision.utils import make_grid, save_image

from tqdm import tqdm
from collections import OrderedDict
import juncw
import shutil
from RIC import Net
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import PerceptualSimilarity.models
from train import getAdvZ, load_checkpoint
from model import load_model_from_checkpoint
from dataset import *

def arg_parser():
    import argparse
    parser = argparse.ArgumentParser(description='RIC encryption')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--data_dir', type=str, default='/data/home/liuchunyu/code/uwf/dr_labeled_junior/train')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--save_dir', type=str, default='DeepUWF/RIC')
    return parser.parse_args()

def main():
    args = arg_parser()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    target_model = load_model_from_checkpoint(
        checkpoint_path = '/data/home/liuchunyu/code/DeepUWF/checkpoints/finetune_drjp_pretrain_ffm_checkpoint-best.pth',
        num_classes = 2,
        device = device,
    )
    target_model = target_model.to(device)
    target_model.eval()

    state_dict = OrderedDict()
    for k, v in torch.load(args.model_path)['state_dict'].items():
        key = k.replace('module.', '')
        state_dict[key] = v
    net = Net()
    net.load_state_dict(state_dict)
    net = net.to(device)
    net.eval()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    dataset = build_dataset(args.data_dir, input_size=args.img_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    for i, (secret, labels) in tqdm(enumerate(dataloader)):
        print('Encrypting ...')
        secret = secret.to(device)
        label = labels.to(device)
        with torch.cuda.amp.autocast():
            cover, succ = getAdvZ(args.img_size, label, args.batch_size, target_model, train=False)
            with torch.no_grad():
                output, recover = net(secret=secret, cover=cover, train=False)
        
        print('Saving ...')
        for j in tqdm(range(args.batch_size)):
            save_dir = os.path.join(args.save_dir, f'class_{labels[j]}')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_image(output[j], os.path.join(save_dir, f'{i * args.batch_size + j}_class_{labels[j]}.jpg'), format='jpeg')


if __name__ == '__main__':
    main()