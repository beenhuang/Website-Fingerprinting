#!/usr/bin/env python3

"""
<file>    dataloader.py
<brief>   
"""

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader


def train_test_dataloader(X, y, test_size=0.2, batch_size=128):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=247, stratify=y)
    print(f"[split] traning size: {len(y_train)}, test size: {len(y_test)}")
 
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(dim=1)
    y_train_tensor = torch.tensor(y_train, dtype=torch.int64)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(dim=1)
    y_test_tensor = torch.tensor(y_test, dtype=torch.int64)

    train_dataloader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=batch_size)    

    return train_dataloader, test_dataloader


if __name__ == '__main__':
    pass
