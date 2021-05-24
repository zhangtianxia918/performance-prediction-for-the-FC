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
feature = 26
batch_size = 16
trainx,testx,trainy,testy = train_test_split(data,label,test_size=0.2)

trainx = torch.from_numpy(trainx[:,0:feature]).type(torch.FloatTensor)
trainy = torch.from_numpy(trainy).type(torch.FloatTensor)

testx = torch.from_numpy(testx[:,0:feature]).reshape(-1,1,feature).type(torch.FloatTensor)
testy = torch.from_numpy(testy).type(torch.FloatTensor)

fctrain = Data.TensorDataset(trainx, trainy)
fctest = Data.TensorDataset(testx, testy)
train_loader = torch.utils.data.DataLoader(dataset=fctrain,batch_size=batch_size,drop_last = True,shuffle=True)
valid_loader = torch.utils.data.DataLoader(dataset=fctrain,batch_size=int(len(fctrain)/5),drop_last = True,shuffle=False)

neurons1 = 1800
dropoutrate = 0.5
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(feature,neurons1),
            nn.Dropout(dropoutrate),
            )
        self.l2 = nn.Sequential(
            nn.Linear(neurons1,neurons1),
            nn.Dropout(dropoutrate),
            )
        self.l3 = nn.Sequential(
            nn.Linear(neurons1,neurons1),
            nn.Dropout(dropoutrate))
        self.l4 = nn.Sequential(
            nn.Linear(neurons1,neurons1),
            nn.Dropout(dropoutrate))
        self.l5 = nn.Sequential(
            nn.Linear(neurons1,neurons1),
            nn.Dropout(dropoutrate))
        
        self.l6 = nn.Sequential(
            nn.Linear(neurons1,21))
        
    def forward(self,x):
        x = x.view(x.shape[0],-1)
        x = F.softsign(self.l1(x))
        x = F.softsign(self.l2(x))
        x = F.softsign(self.l3(x))
        x = F.softsign(self.l4(x))
        x = F.softsign(self.l5(x))
        x = self.l6(x)
        return x

    
def Train():
    
    running_loss = .0
    val_loss = .0
    
    train_e = int(len(fctrain)/batch_size)-3
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
    val_loss = val_loss/(3)
    return train_loss,val_loss



        
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Model().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001,weight_decay=0.001)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.1,weight_decay=0.01)
criterion = nn.MSELoss()

atrain_losses = []
avalid_losses = []
ae = []
epochs = 501
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

torch.save(model.state_dict(),'dnn_predict.pth')
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









