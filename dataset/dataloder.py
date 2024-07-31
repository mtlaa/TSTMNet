import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from torchvision.transforms import transforms
import cv2
import torch




class HDRDataset(Dataset):
    def __init__(self, annotations_file, data_root_dir, hdr_transform=None, ldr_transform=None):
        self.img_infos = pd.read_csv(annotations_file)
        self.root_dir = data_root_dir
        self.hdr_transform = hdr_transform
        self.ldr_transform = ldr_transform
    
    def __len__(self):
        return len(self.img_infos)
    
    def __getitem__(self, index):

        hdr_path = self.root_dir + self.img_infos.iloc[index][0]
        ldr_path = self.root_dir + self.img_infos.iloc[index][1]
        hdr = cv2.imread(hdr_path, cv2.IMREAD_ANYDEPTH)   
        ldr = cv2.imread(ldr_path)

        if hdr.shape != (512, 512, 3):
            hdr = cv2.resize(hdr, (512, 512))
        if ldr.shape != (512, 512, 3):
            ldr = cv2.resize(ldr, (512, 512))

        hdr = cv2.cvtColor(hdr, cv2.COLOR_RGB2HSV)   
        ldr = cv2.cvtColor(ldr, cv2.COLOR_RGB2HSV)
        hdr = torch.permute(torch.from_numpy(hdr), (2, 0, 1))[2:] + 0.000001 
        ldr = torch.permute(torch.from_numpy(ldr), (2, 0, 1))[2:]

        hdr = torch.log(hdr)
        hdr = (hdr-hdr.min())/(hdr.max()-hdr.min())
        ldr = ldr/255.
        
        if self.hdr_transform:
            hdr = self.hdr_transform(hdr)
        if self.ldr_transform:
            ldr = self.ldr_transform(ldr)
        return hdr, ldr


class TestDataset(Dataset):
    def __init__(self, annotations_file, data_root_dir, transform=None):
        self.img_infos = pd.read_csv(annotations_file)
        self.ldr_img_infos = pd.read_csv('dataset/test_ldr_new.csv')
        self.root_dir = data_root_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_infos)
    
    def __getitem__(self, index):
        hdr_path = self.root_dir + self.img_infos.iloc[index][0]
        ldr_path = self.root_dir + self.ldr_img_infos.iloc[index][0]
        ldr = cv2.imread(ldr_path)
        hdr = cv2.imread(hdr_path, cv2.IMREAD_ANYDEPTH)
        raw_hdr = torch.permute(torch.from_numpy(hdr), (2, 0, 1))
        hdr = cv2.cvtColor(hdr, cv2.COLOR_RGB2HSV)
        hdr = torch.permute(torch.from_numpy(hdr), (2, 0, 1))
        hdr_v = hdr[2:] + 0.000001
        hdr_v = torch.log(hdr_v)
        hdr_v = (hdr_v-hdr_v.min())/(hdr_v.max()-hdr_v.min())
        if self.transform:
            hdr_v = self.transform(hdr_v)
            hdr = self.transform(hdr)
            raw_hdr = self.transform(raw_hdr)
        return hdr_v, hdr, raw_hdr, ldr


def get_train_dataloader(batch_size, num_works, hdr_transform=None, ldr_transform=None):
    annotations_file = 'dataset/train_data.csv'  
    data_root_dir = '/home/mtlaa/project_dl/data/train_512/'

    
    hdrdataset = HDRDataset(annotations_file, data_root_dir, hdr_transform=hdr_transform, ldr_transform=ldr_transform)

    dataloader = DataLoader(hdrdataset, batch_size=batch_size, num_workers=num_works, shuffle=True, drop_last=True)
    return dataloader


def get_test_dataloader(batch_size=1, num_works=4, transform=None):
    annotations_file = 'dataset/test_hdr_new.csv'  

    data_root_dir = '/home/mtlaa/project_dl/data/'

    dataset = TestDataset(annotations_file, data_root_dir, transform)
    dataloader = DataLoader(dataset, batch_size, num_workers=num_works)
    return dataloader