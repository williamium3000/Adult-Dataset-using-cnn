import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset

train_data = np.load("preprocessed_data/train_data.npy", allow_pickle=True)
train_label = np.load("preprocessed_data/train_label.npy", allow_pickle=True)
test_data = np.load("preprocessed_data/test_data.npy", allow_pickle=True)
test_label = np.load("preprocessed_data/test_label.npy", allow_pickle=True)

class adultDataset(Dataset):
    def __init__(self, data, label):
        self.data = data[:, np.newaxis, :]
        self.label = label
        
        
        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, item):
        return self.data[item], self.label[item]

datasets = {"train": adultDataset(train_data, train_label), "val": adultDataset(test_data, test_label)}
phase = ["train", "val"]
def get_dataLoaders():
    return {x: torch.utils.data.DataLoader(datasets[x], batch_size=64,
                                                            shuffle=True, num_workers=0, pin_memory=True, 
                                                            drop_last=True)
                            for x in phase}
                        

if __name__ == "__main__":
    print(get_dataLoaders())