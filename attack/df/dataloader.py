#!/usr/bin/env python3

"""
<file>    dataloader.py
<brief>   
"""

import torch
import random
from collections import Counter
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader


def train_valid_test_dataloader(X, y, valid_size=0.1, test_size=0.1, batch_size=128):
    X_train, X_valid_test, y_train, y_valid_test = train_test_split(X, y, test_size=valid_size+test_size, random_state=random.randint(0,10000), stratify=y)
    X_valid, X_test, y_valid, y_test = train_test_split(X_valid_test, y_valid_test, test_size=test_size/(valid_size+test_size), random_state=random.randint(1,10000), stratify=y_valid_test)
    print(f"[split size] tran:{len(y_train)}, valid:{len(y_valid)}, test:{len(y_test)}")
    print(f"[label] train:{Counter(y_train)}, valid:{Counter(y_valid)}, test:{Counter(y_test)}") 
        
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(dim=1)
    y_train_tensor = torch.tensor(y_train, dtype=torch.int64)
    X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32).unsqueeze(dim=1)
    y_valid_tensor = torch.tensor(y_valid, dtype=torch.int64)    
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(dim=1)
    y_test_tensor = torch.tensor(y_test, dtype=torch.int64)
    
    print(f"[data shape] tran:{X_train_tensor.shape}, valid:{X_valid_tensor.shape}, test:{X_test_tensor.shape}")

    train_dataloader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
  
    return train_dataloader, (X_valid_tensor, y_valid_tensor), (X_test_tensor, y_test_tensor)


def train_test_dataloader(X, y, test_size=0.2, batch_size=128):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random.randint(0,10000), stratify=y)
    print(f"[split size] tran:{len(y_train)}, test:{len(y_test)}")
    print(f"[label] train:{Counter(y_train)}, test:{Counter(y_test)}") 
         
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(dim=1)
    y_train_tensor = torch.tensor(y_train, dtype=torch.int64)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(dim=1)
    y_test_tensor = torch.tensor(y_test, dtype=torch.int64)

    train_dl = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)

    print(f"[data shape] tran:{X_train_tensor.shape}, test:{X_test_tensor.shape}")

    return train_dl, (X_test_tensor, y_test_tensor)


if __name__ == '__main__':
    import pickle
    with open("/Users/huangbin/desktop/WF/script/attack/varcnn/feature/Wang-20000/feature.pkl", "rb") as f:
        X, y = pickle.load(f)

    train_dataloader, valid_data, test_data = train_valid_test_dataloader(X, y)

