# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 22:20:15 2020

@author: zhangtianxia
"""
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
from sklearn.model_selection import train_test_split



data = np.load('data.npy')
label = np.load('label.npy')
feature = 20
epochs = 501
batch_size = 16
trainx,testx,trainy,testy = train_test_split(data,label,test_size=0.1)

# trainx = torch.from_numpy(trainx[:,0:feature]).reshape(-1,1,5,int(feature/5)).type(torch.FloatTensor)
# trainy = torch.from_numpy(trainy).type(torch.FloatTensor)
trainx = torch.from_numpy(data[:,0:feature]).reshape(-1,1,5,int(feature/5)).type(torch.FloatTensor)
trainy = torch.from_numpy(label).type(torch.FloatTensor)

testx = torch.from_numpy(testx[:,0:feature]).reshape(-1,1,5,int(feature/5)).type(torch.FloatTensor)
testy = torch.from_numpy(testy).type(torch.FloatTensor)

fctrain = Data.TensorDataset(trainx, trainy)
fctest = Data.TensorDataset(testx, testy)
train_loader = torch.utils.data.DataLoader(dataset=fctrain,batch_size=batch_size,drop_last = True,shuffle=True)
valid_loader = torch.utils.data.DataLoader(dataset=fctrain,batch_size=int(len(fctrain)/5),drop_last = True,shuffle=False)

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16)
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32)
            )
        self.fc1 = nn.Linear(640,64)
        self.fc2 = nn.Linear(64,21)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.fc1(x))
        x = F.relu(x)
        # x = F.softsign(x)
        x = self.fc2(x)
        return x
    
def Train():
    
    running_loss = .0
    val_loss = .0
    
    train_e = int(len(fctrain)/batch_size)-2
    for idx, (inputs,labels) in enumerate(train_loader):
        model.train()
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        preds = model(inputs.float())
        loss = criterion(preds,labels)
        loss.backward()
        optimizer.step()
        model.eval() 
        if idx<(train_e):

            running_loss += criterion(model(inputs.float()),labels)
        else:
            val_loss += criterion(model(inputs.float()),labels)
        
    model.eval()    
    train_loss = running_loss/(train_e)
    val_loss = val_loss/(2)
    return train_loss,val_loss


def Valid():
    running_loss = .0
    
    model.eval()
    
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(valid_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            preds = model(inputs.float())
            loss = criterion(preds,labels)
            running_loss += loss
            
        valid_loss = running_loss/len(valid_loader)
    return valid_loss
        
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Model().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001,weight_decay=0.001)
criterion = nn.MSELoss()
atrain_losses = []
avalid_losses = []
ae = []

for epoch in range(epochs):
    
    train_loss,val_loss = Train()
    if epoch%50 ==0:
        ae.append(epoch)
        atrain_losses.append(train_loss.cpu().detach().numpy())
        print('epochs {}/{}'.format(epoch,epochs))
        print(f'train_loss {train_loss}')
        avalid_losses.append(val_loss.cpu().detach().numpy())
        print(f'valid_loss {val_loss}')
ae = np.array(ae)
atrain_losses = np.array(atrain_losses)
avalid_losses = np.array(avalid_losses)

torch.save(model.state_dict(),'cnn_predict.pth')
model.eval()

data,label = iter(valid_loader).next()

output = model(data.cuda()).data.cpu().numpy()

label = label.data.cpu().numpy()  

x = range(len(label[0]))
import matplotlib.pyplot as plt 
plt.figure(figsize=(6, 4))
plt.plot(x,label[0],
           linewidth = '2', color='royalblue',
           label='prediction-train')
plt.plot(x,output[0],
           linewidth = '2', color='orangered',
           label='prediction-train')







