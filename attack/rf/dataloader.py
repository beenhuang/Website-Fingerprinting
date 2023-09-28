#!/usr/bin/env python3

"""
<file>    dataloader.py
<brief>   
"""

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, TensorDataset

def train_test_dataloader2(X, y, test_size=0.2, batch_size=200):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=247, stratify=y)

    train_dataloader = DataLoader(MyDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(MyDataset(X_test, y_test), batch_size=batch_size)    

    print(f"[split] traning size: {len(y_train)}, test size: {len(y_test)}")
 
    return train_dataloader, test_dataloader

def train_test_dataloader(X, y, test_size=0.2, batch_size=200):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=247, stratify=y)

    train_dataloader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32).unsqueeze(dim=1), torch.tensor(y_train)), batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32).unsqueeze(dim=1), torch.tensor(y_test)), batch_size=batch_size)    

    print(f"[split] traning size: {len(y_train)}, test size: {len(y_test)}")
 
    return train_dataloader, test_dataloader

class MyDataset(Dataset):
    def __init__(self, X, y):
        self.data = torch.tensor(X, dtype=torch.float32).unsqueeze(dim=1)
        self.label = torch.tensor(y)
        
    def __getitem__(self, idx):
        d = self.data[idx]
        l = self.label[idx]

        return d,l
    
    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    pass
