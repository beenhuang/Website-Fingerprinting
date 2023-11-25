#!/usr/bin/env python3

"""
<file>    dfnet.py
<brief>   DF model with PyTorch
"""

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from torchsummary import summary

from metrics import *

class VarClassifier():
    def __init__(self, classes, device, m_file=None):
        self.device = device
        self.classes = classes
        self.model_dir = VarNet(classes).to(device) # two varnet models
        self.model_time = VarNet(classes).to(device)  
        self.optimizer_dir = Adam(self.model_dir.parameters())
        self.scheduler_dir = ReduceLROnPlateau(self.optimizer_dir, mode='min', factor=np.sqrt(0.1), patience=5, min_lr=1e-5)
        self.optimizer_time = Adam(self.model_time.parameters())      
        self.scheduler_time = ReduceLROnPlateau(self.optimizer_time, mode='min', factor=np.sqrt(0.1), patience=5, min_lr=1e-5)
        self.loss_fn = nn.CrossEntropyLoss()
        self.n_epochs = 150 # default is 150

        if m_file != None:
            checkpoint = torch.load(m_file)
            self.model_dir.load_state_dict(checkpoint['model_dir'])
            self.model_time.load_state_dict(checkpoint['model_time'])
            self.threshold = checkpoint['threshold']

    def train(self, train_dl, valid_data):
        self.__train_one_varnet(train_dl, valid_data, self.model_dir, self.optimizer_dir, self.scheduler_dir, 'dir')
        self.__train_one_varnet(train_dl, valid_data, self.model_time, self.optimizer_time, self.scheduler_time, 'time')
        self.threshold = self.__get_threshold(valid_data)
        
    def __train_one_varnet(self, train_dl, valid_data, model, optimizer, scheduler, m_name): 
        # Early stopping
        last_acc = 100
        patience = 10 # original set is 10
        trigger_times = 0  

        for epoch in range(self.n_epochs):
            self.__train_one_epoch(epoch+1, train_dl, model, optimizer, m_name)
            current_loss, current_acc = self.__validate(epoch+1, valid_data, model, m_name)
            scheduler.step(current_loss) 

            # Early stopping
            if current_acc < last_acc:
                trigger_times += 1
                print('Trigger Times:', trigger_times)

                if trigger_times >= patience:
                    print(f'Early stopping!')
                    break
            else:
                trigger_times = 0

            last_acc = current_acc        
                    
    def __train_one_epoch(self, epoch_idx, train_dl, model, optimizer, m_name):
        model.train() # update batch_normalization and enable dropout layer
        running_loss = 0.0 # loss value

        for batch_X, batch_y in train_dl:
            if m_name == 'dir':
                batch_X = batch_X[:,:,:5007]
            else:
                batch_X = batch_X[:,:,5007:]   
            # dataset load to device
            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

            # train
            optimizer.zero_grad() # Zero your gradients for every batch!
            pred = model(batch_X) # Make predictions for this batch
            loss = self.loss_fn(pred, batch_y) # Compute the loss and its gradients
            loss.backward()
            optimizer.step() # Adjust learning weights

            running_loss += loss.item() # accumulate loss value
        
        print(f"[{m_name}: Epoch_{epoch_idx}] Avg_loss:{running_loss/len(train_dl):.4f} (total_loss/num_batch):{running_loss:.4f}/{len(train_dl)}")  

    def __validate(self, epoch_idx, valid_data, model, m_name):
        X_valid, y_valid = valid_data[0], valid_data[1]
        model.eval() # Set the model to evaluation mode, disabling dropout and using population statistics for batch normalization.
     
        with torch.no_grad(): # Disable gradient computation and reduce memory consumption.
            if m_name == 'dir':
                X_valid = X_valid[:,:,:5007]
            else:
                X_valid = X_valid[:,:,5007:] 
            # dataset load to device
            X_valid, y_valid = X_valid.to(self.device), y_valid.to(self.device)
                
            pred = model(X_valid) # Make predictions for this batch
            vloss = self.loss_fn(pred, y_valid) # Compute the loss and its gradients

            y_pred = [np.argmax(x) for x in F.softmax(pred, dim=1).data.cpu().tolist()]
            acc = sum([1 if x[0] == x[1] else 0 for x in zip(y_valid,y_pred)]) / float(len(y_pred))

        print(f"[{m_name}: Epoch_{epoch_idx}_valid] Total_loss:{vloss:.4f}, Acc:{acc:.5f}")  

        return vloss, acc

    def __get_threshold(self, valid_data):
        X_valid, y_valid = valid_data[0], valid_data[1]
        self.model_dir.eval()
        self.model_time.eval()

        with torch.no_grad(): # Disable gradient computation and reduce memory consumption.
            X_valid_dir, X_valid_time = X_valid[:,:,:5007], X_valid[:,:,5007:]
            X_valid_dir, X_valid_time, y_valid = X_valid_dir.to(self.device), X_valid_time.to(self.device), y_valid.to(self.device)
            
            pred_direct = self.model_dir(X_valid_dir) # Make predictions for this batch
            pred_time = self.model_time(X_valid_dir)

            pred_direct = F.softmax(pred_direct, dim=1).data.cpu()
            pred_time = F.softmax(pred_time, dim=1).data.cpu()
            y_pred_score = [(np.argmax(x), np.max(x)) for x in torch.div(torch.add(pred_direct, pred_time), 2).tolist()]
    
        rec_th = []
        for th in np.arange(0, 1.01, 0.1):
            rec = self.__recall_with_threshold(y_valid, y_pred_score, th)
            rec_th.append(rec)

        best_th = rec_th.index(max(rec_th)) * 0.1 

        #print(f"y_pred_score:{y_pred_score}")
        #print(f"threshold results:{rec_th}")
        #print(f"best threshold:{best_th}, max recall: {max(rec_th)}")

        return best_th

    def __recall_with_threshold(self, y_true, pred_score, th, label_unmon=0):  
        y_pred = []

        # if the sample's probability is less than the threshold, the sample will get the unmonitord label.
        for x in pred_score:
            pred, score = x[0], x[1]

            if score < th: # prob < threshold
                y_pred.append(label_unmon) # get the unmonitord label.
            else:
                y_pred.append(pred) # get the prediction label
        
        if self.classes == 2:
            res = ow_score_twoclass(y_true, y_pred, return_str=False)
        else:
            res =  ow_score_multiclass(y_true, y_pred, return_str=False)   
        
        return res

    def test(self, test_data, label_unmon=0):
        X_test, y_test = test_data[0], test_data[1]
        self.model_dir.eval() # not update batch_normalization and disable dropout layer
        self.model_time.eval()
        
        with torch.no_grad(): # Disable gradient computation and reduce memory consumption.
            X_test_dir, X_test_time = X_test[:,:,:5007], X_test[:,:,5007:]
            X_test_dir, X_test_time, y_test = X_test_dir.to(self.device), X_test_time.to(self.device), y_test.to(self.device)
              
            pred_direct = self.model_dir(X_test_dir) # Make predictions for this batch
            pred_time = self.model_time(X_test_time)

            pred_direct = F.softmax(pred_direct, dim=1).data.cpu()
            pred_time = F.softmax(pred_time, dim=1).data.cpu()
            pred = torch.div(torch.add(pred_direct, pred_time), 2).tolist()
            y_pred = [np.argmax(x) if np.max(x)>self.threshold else label_unmon for x in pred]
            #y_pred = [np.argmax(x) if np.max(x)>self.threshold else label_unmon for x in pred]
    
            if self.classes == 2:
                pos_score = [x[1] for x in pred]
            else:
                pos_score = []
                
        return y_test.tolist(), y_pred, pos_score

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, dialtion):
        super(ConvBlock, self).__init__() 
        self.conv_branch_2a=nn.Conv1d(in_channels, out_channels, 3, stride=stride, padding=1, dilation=dialtion[0], bias=False)
        self.batchnorm_branch_2a=nn.BatchNorm1d(out_channels)
        self.conv_branch_2b=nn.Conv1d(out_channels, out_channels, 3, padding=2, dilation=dialtion[1], bias=False)
        self.batchnorm_branch_2b=nn.BatchNorm1d(out_channels)
        self.conv_branch_1=nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False)
        self.batchnorm_branch_1=nn.BatchNorm1d(out_channels)

    
    def forward(self, x):
        identity = x

        x = self.conv_branch_2a(x)
        x = self.batchnorm_branch_2a(x)
        x = F.relu(x)
        x = self.conv_branch_2b(x)
        x = self.batchnorm_branch_2b(x)

        identity = self.conv_branch_1(identity)
        identity = self.batchnorm_branch_1(identity)

        x += identity
        x = F.relu(x)

        return x

class IdentityBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, dialtion):
        super(IdentityBlock, self).__init__() 
        self.conv_branch_2a=nn.Conv1d(in_channels, out_channels, 3, stride=stride, padding=4, dilation=dialtion[0], bias=False)
        self.batchnorm_branch_2a=nn.BatchNorm1d(out_channels)
        self.conv_branch_2b=nn.Conv1d(out_channels, out_channels, 3, padding=8, dilation=dialtion[1], bias=False)
        self.batchnorm_branch_2b=nn.BatchNorm1d(out_channels)
   
    
    def forward(self, x):
        identity = x

        x = self.conv_branch_2a(x)
        x = self.batchnorm_branch_2a(x)
        x = F.relu(x)
        x = self.conv_branch_2b(x)
        x = self.batchnorm_branch_2b(x)

        x += identity
        x = F.relu(x)

        return x
      
class ResNet18(nn.Module):
    
    def __init__(self, classes):
        super(ResNet18, self).__init__()
        self.conv1 = self.__conv1_block(1, 64)
        self.conv2_a = ConvBlock(64, 64, 1, (1,2))
        self.conv2_b = IdentityBlock(64, 64, 1, (4,8))
        self.conv3_a = ConvBlock(64, 128, 2, (1,2))
        self.conv3_b = IdentityBlock(128, 128, 1, (4,8))
        self.conv4_a = ConvBlock(128, 256, 2, (1,2))
        self.conv4_b = IdentityBlock(256, 256, 1, (4,8))
        self.conv5_a = ConvBlock(256, 512, 2, (1,2))
        self.conv5_b = IdentityBlock(512, 512, 1, (4,8))      

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_a(x)
        x = self.conv2_b(x)
        x = self.conv3_a(x)
        x = self.conv3_b(x)
        x = self.conv4_a(x)
        x = self.conv4_b(x)
        x = self.conv5_a(x)
        x = self.conv5_b(x)
        x = F.adaptive_avg_pool1d(x, 1)

        return x        

    # convolutional block
    def __conv1_block(self, channels_in, channels_out):
        return nn.Sequential(nn.Conv1d(channels_in, channels_out, kernel_size=7, stride=2, padding=3, bias=False),
                             nn.BatchNorm1d(channels_out),
                             nn.ReLU(),
                             nn.MaxPool1d(3, stride=2, padding=1))

class VarNet(nn.Module):
    
    def __init__(self, classes):
        super(VarNet, self).__init__()
        self.resnet_block = ResNet18(classes)
        self.meta_block = self.__metadata_block(7, 32)
        self.fc_block = self.__fc_block(512+32, classes)

    def forward(self, x):
        res_input = x[:, :, :5000]
        meta_input = x[:, :, 5000:].squeeze(dim=1) # remove channel: (N, C, L) -> (N, L)

        res_output = self.resnet_block(res_input)
        meta_output = self.meta_block(meta_input)
        
        res_meta = torch.cat((res_output.squeeze(dim=2), meta_output), dim=1)

        out = self.fc_block(res_meta)

        return out

    def __metadata_block(self, in_features, out_features):    
        return nn.Sequential(nn.Linear(in_features, out_features),
                             nn.BatchNorm1d(out_features),
                             nn.ReLU())

    def __fc_block(self, in_features, classes):
        return nn.Sequential(nn.Linear(in_features, 1024),
                             nn.BatchNorm1d(1024),
                             nn.ReLU(),
                             nn.Dropout(0.5),
                             nn.Linear(1024, classes))
                             #nn.Softmax(dim=1))     
    
if __name__ == '__main__':
    #model = ResNet18(100)
    #output = model(torch.rand(2, 1, 5000))    
    #print(summary(model, (1, 5000)))  
    #print(output.shape) 

    model = VarNet(100)
    output = model(torch.rand(2, 1, 5007))    
    print(summary(model, (1, 5007)))  
    print(output.shape) 
