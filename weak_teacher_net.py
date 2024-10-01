import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import os
join = os.path.join
from PIL import Image
from tqdm import tqdm
import pandas as pd
import random

from Model.models_vit import resnet50 as resnet50
from Model.util.datasets import build_transform
from arguments import get_args_parser

from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

def split_dataset(data_dir, label_csv_path, args, proportion=0.8):
    train_transform = build_transform('train', args)
    test_transform = build_transform('test', args)
    dataset = pd.read_csv(label_csv_path)
    
    train_size = int( proportion * len(dataset) )
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    train_samples = dataset[: train_size]
    test_samples = dataset[train_size :]
    
    train_dataset = LabeledDataset.fromarray(train_samples, train_transform, data_dir=data_dir)
    test_dataset = LabeledDataset.fromarray(test_samples, test_transform, data_dir=data_dir)
    print('Build training dataset with size {} and test dataset with size {}'.format(
        len(train_dataset), len(test_dataset)
    ))
    return train_dataset, test_dataset


arg_parser = get_args_parser()
args = arg_parser.parse_args()

root = '/data/home/liuchunyu/code/uwf/japan'
transform = build_transform(True, args)
label_csv = '/data/home/liuchunyu/code/uwf/japan_uwf_data.csv'


class LabeledDataset(Dataset):
    def __init__(self, data_dir, label_csv_path=None, is_train=None, args=None, **kwargs):
        '''
        used for dataset with .pkl labels in the following format:
        [
            [img_file_name: str, label: int],
            ...
        ]
        '''
        super(Dataset, self).__init__(**kwargs)
        self.root = data_dir
        if label_csv_path==None:
            return
        self.label_df = pd.read_csv(label_csv_path)
        self.img_paths = self.label_df['filename'].tolist()
        # initialize transform
        if is_train is None or args is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = build_transform(is_train, args)
    
    @classmethod
    def fromarray(cls, df, transform=None, **kwargs):
        '''
        Build from array
        '''
        obj = cls(**kwargs)
        obj.img_paths = df['filename'].tolist()
        obj.label_df = df
        # initialize transform
        if transform is None:
            obj.transform = transforms.ToTensor()
        else:
            obj.transform = transform
        return obj

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        img_path = join(self.root, self.img_paths[index])
        img = Image.open(img_path)
        img = self.transform(img)
        label = self.label_df.iloc[index, 5:13].to_list()
        if sum(label) == 0:
            label = [1] + label
        else:
            label = [0] + label
        return img, torch.tensor(label, dtype=torch.float32)
    
is_train = False
    
mydt = LabeledDataset(root, label_csv, is_train=is_train)
sampler = torch.utils.data.RandomSampler(mydt)
data_loader = torch.utils.data.DataLoader(
            mydt,
            sampler=sampler,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )

net = resnet50(args).to(device=args.device) # 1 normal and 8 diseases


if is_train==False:
    # load weights
    checkpoint = torch.load('/data/home/liuchunyu/code/uwf/jp_resnet50_model.pth', map_location='cpu')
    net.load_state_dict(checkpoint)

    net.train(False)
    acf = nn.Sigmoid()
    acf.train(False)
    
    true_label_decode_list = []
    pred_label_decode_list = []

    for i, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to(device=args.device), targets.to(device=args.device)
        outputs = net(inputs)
        outputs = acf(outputs)
        # calculate auc
        auc = roc_auc_score(targets.cpu().numpy(), outputs.cpu().detach().numpy(),
                            average='samples', multi_class='ovr')
        print(outputs.cpu().detach().numpy(), targets.cpu().numpy())
        print(f'batch {i}, auc: {auc}')

    
    
else:
        
    train_dataset, test_dataset = split_dataset(root, label_csv, args)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size*2,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )


    # begin to train
    optimizer = torch.optim.Adam(net.parameters(), lr=2e-4)
    # loss function for multi-class
    criterion = nn.BCELoss()

    max_auc = 0

    for epoch in range(300):
        
        true_label_decode_list = []
        pred_label_decode_list = []
        loss_sum = 0
        
        net.train(True)
        for i, (inputs, targets) in tqdm(enumerate(train_dataloader)):
            inputs, targets = inputs.to(device=args.device), targets.to(device=args.device)
            optimizer.zero_grad()
            outputs = net(inputs)
            outputs = nn.Sigmoid()(outputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            
            true_label_decode_list.extend(targets.cpu().numpy())
            pred_label_decode_list.extend(outputs.cpu().detach().numpy())
        # calculate auc
        auc_train = roc_auc_score(true_label_decode_list, pred_label_decode_list,
                            average='samples', multi_class='ovr')
        print(f'Epoch {epoch}, train auc: {auc_train}, loss: {loss_sum/len(data_loader)}')
        
        
        # test
        true_label_decode_list = []
        pred_label_decode_list = []
        loss_sum = 0
        
        net.train(False)
        for i, (inputs, targets) in tqdm(enumerate(test_dataloader)):
            inputs, targets = inputs.to(device=args.device), targets.to(device=args.device)
            outputs = net(inputs)
            outputs = nn.Sigmoid()(outputs)
            loss = criterion(outputs, targets)
            loss_sum += loss.item()
            
            true_label_decode_list.extend(targets.cpu().numpy())
            pred_label_decode_list.extend(outputs.cpu().detach().numpy())
        # calculate auc
        auc_test = roc_auc_score(true_label_decode_list, pred_label_decode_list,
                            average='samples', multi_class='ovr')
        
        print(f'Epoch {epoch}, test auc: {auc_test}, loss: {loss_sum/len(data_loader)}')
        
        if auc_test > max_auc:
            max_auc = auc_test
            torch.save(net.state_dict(), '/data/home/liuchunyu/code/uwf/jp_resnet50_model_300.pth')
            print('Model saved')
