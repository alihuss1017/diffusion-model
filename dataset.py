import os
import torch
from torch.utils.data import Dataset, Subset, random_split, DataLoader
from torchvision.io import decode_image

class MyDataset(Dataset):
    def __init__(self, img_dir: str):
        self.img_dir = img_dir 
    
    def __len__(self):
        return len(os.listdir(self.img_dir))

    def __getitem__(self, idx):
        return decode_image(f'data/{os.listdir(self.img_dir)[idx]}')
    

class BuildLoaders():
    def __init__(self, train_split: float, val_split: float, 
                batch_size: int, subset_size: int | None = None):
    
        self.subset_size = subset_size
        self.train_split = train_split
        self.val_split = val_split
        self.batch_size = batch_size

        self.dataset = MyDataset('data')

        self.train_data = None
        self.val_data = None
        self.train_loader = None
        self.val_loader = None

    def _create_subset(self):
        if self.subset_size:
            indices = torch.randint(0, len(self.dataset), (self.subset_size, 1))
            self.dataset = Subset(self.dataset, indices)
    
    def _create_train_val_loaders(self):
        self.train_data, self.val_data = random_split(self.dataset, 
                                                      [self.train_split, self.val_split])
        
        self.train_loader = DataLoader(self.train_data, batch_size = self.batch_size,
                                       drop_last = True)
        self.val_loader = DataLoader(self.val_data, batch_size = self.batch_size, 
                                     drop_last = True)
        
    def build(self):
        '''returns train and val dataloaders.'''
        self._create_subset()
        self._create_train_val_loaders()

        return self.train_loader, self.val_loader

