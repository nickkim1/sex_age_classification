import csv
import torch 
import torch.nn as nn
import torch.nn.functional as F
import random
from sklearn import metrics
from torch.utils.data import Dataset, DataLoader, BatchSampler
import torch.utils.data as torch_data
from matplotlib import pyplot as plt
import seaborn as sn
import pandas as pd 
import numpy as np
import time as time
import os
import argparse as ap
import data_preprocessing as p

### NOTE ### 
# 1. this is not the full implementation of the Basset architecture
# 2. no need to preprocess because this is just bulk data, not single cell (can just use raw counts)

### COMMANDS FOR IMPLEMENTATION ### 
###################################

# first set the device used
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("===================================")
print(f"0. Using {device} device!")

### real dataset for use ### 
class CustomDataset(Dataset): 
    def __init__(self, dataset, n_samples): 
        # xy = np.loadtxt(file_path, delimiter=",", dtype=np.float32) # data loaded in will be comma delimited
        self.n_samples = n_samples
        self.features = np.empty([self.n_samples, 600, 4]) # (13854x1) 
        self.labels = np.empty([self.n_samples, 1])
        for i in range(self.n_samples): # n_samples is the number of rows, 13854
            s = np.zeros([600,4]) # <- (13854x600)x4
            matching_row_sequence = dataset[i][1]
            matching_row_label = dataset[i][2]
            for j in range(600): 
                # one hot encode sequences as: A T C G (top -> bottom)
                idx_to_mark = 0
                if matching_row_sequence[j] == 'A': 
                    idx_to_mark = 0
                elif matching_row_sequence[j] == 'C':
                    idx_to_mark = 1
                elif matching_row_sequence[j] == 'T': 
                    idx_to_mark = 2
                elif matching_row_sequence[j] == 'G':
                    idx_to_mark = 3
                s[j][idx_to_mark] = 1
            self.features[i] = s # set the sequence @ the range of a given gene -> the specified seequence 
            self.labels[i] = matching_row_label
        self.features = torch.from_numpy(self.features).float() 
        self.gene_ids = torch.from_numpy(np.arange(0, self.n_samples, 1))
        self.classes = list(self.gene_ids)
    def __len__(self):
        return self.n_samples
    def __getitem__(self, index):
        return self.features[index], self.labels[index]
    def gene_ids_and_indices(self):
        return self.classes, self.features

### custom batch sampler class (for generating the 4 x 600 inputs) etc ###
class BSampler(BatchSampler): 
    def __init__(self, num_rows, gene_ids, indices, batch_size):
        super(BSampler, self).__init__(train_dataset, batch_size, drop_last=False) # i forget why dataset is needed here 
        self.gene_ids = gene_ids
        self.num_rows = num_rows
        self.indices = indices 
    def __iter__(self):
        batches = []
        for ignore in range(self.num_rows):
            batch = [random.choice(self.gene_ids)] # randomly choose an idx (sample), from 0 to 13584
            batches.append(batch) # append each batch to the batches list
        return iter(batches)
    def __len__(self):
        # this doesn't return anything informative unless i change the num_rows into constructor param
        return self.num_rows

# train dataset 
training, testing, validation = p.training, p.testing, p.validation
train_dataset = CustomDataset(dataset=training, n_samples=np.shape(training)[0])
train_dataloader = DataLoader(dataset=train_dataset, batch_sampler=BSampler(num_rows=len(train_dataset), gene_ids=train_dataset.gene_ids_and_indices()[0], indices=train_dataset.gene_ids_and_indices()[1], batch_size=1))
train_feat, train_label = next(iter(train_dataloader))
# print(train_feat, train_label)
n_total_steps = len(train_dataloader) 
# print(n_total_steps)

# test dataset 
test_dataset = CustomDataset(dataset=testing, n_samples=np.shape(testing)[0])
test_dataloader = DataLoader(dataset=test_dataset, batch_sampler=BSampler(num_rows=len(test_dataset), gene_ids=test_dataset.gene_ids_and_indices()[0], indices=test_dataset.gene_ids_and_indices()[1], batch_size=1))
test_feat, test_label = next(iter(test_dataloader))

### define the CNN architecture for basset ###
class BassetCNN(nn.Module):
    def __init__(self): 
        super(BassetCNN, self).__init__()
        self.conv_one = nn.Conv1d(in_channels=4, out_channels=300, kernel_size=19)
        self.batchnorm_one = nn.BatchNorm1d(582) # input = num channels 
        self.pool_one = nn.MaxPool1d(3) 
        self.conv_two = nn.Conv1d(in_channels=300, out_channels=200, kernel_size=11)
        self.batchnorm_two = nn.BatchNorm1d(184) # input = num channels 
        self.pool_two = nn.MaxPool1d(4)
        self.conv_three = nn.Conv1d(in_channels=200, out_channels=200, kernel_size=7)
        self.batchnorm_three = nn.BatchNorm1d(40) # input = num channels 
        self.pool_three = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2000, 1000) # 1000 unit linear layer (same as paper)
        self.dropout_one = nn.Dropout1d(0.3) # for some reason the 1D dropout doesn't work  
        self.fc2 = nn.Linear(1000, 1000) # unsure of specific rationale why they kept the layer the same size, guess it was just optimal? 
        self.dropout_two = nn.Dropout1d(0.3)
        self.fc3 = nn.Linear(1000, 164) # output dim should be [1x164] since i unsqueezed @ flattening above
    def forward(self, x): 
        x = self.conv_one(x)
        # print('first conv layer: ', x.shape)
        x = self.batchnorm_one(x)
        # print('first batchnorm layer: ', x.shape)
        x = F.relu(x)
        # print('first relu: ', x.shape)
        x = self.pool_one(x)
        # print('first pooling: ', x.shape)
        x = self.conv_two(x)
        # print('second conv layer: ', x.shape)
        x = self.batchnorm_two(x)
        # print('second batchnorm layer: ', x.shape)
        x = F.relu(x)
        # print('second relu: ', x.shape)
        x = self.pool_two(x)
        # print('second pooling: ', x.shape)
        x = self.conv_three(x)
        # print('third conv layer: ', x.shape)
        x = self.batchnorm_three(x)
        # print('third batchnorm layer: ', x.shape)
        x = F.relu(x)
        # print('third relu ', x.shape)
        x = self.pool_three(x)
        # print('third pooling: ', x.shape)
        x = x.view(2000) # view for the output layer -- should just be the product of the first and escond dimension
        # print('flattened layer: ', x.shape)
        x = self.fc1(x.unsqueeze(0)) # unsqueeze after flattening
        # print('first fc layer: ', x.shape)
        x = F.relu(x)
        # print('fourth relu ', x.shape)
        x = self.dropout_one(x)
        # print('first dropout ', x.shape)
        x = self.fc2(x)
        # print('second fc layer: ', x.shape)
        x = self.dropout_two(x)
        # print('second dropout layer: ', x.shape)
        x = self.fc3(x)
        # print('final fc layer: ', x.shape)
        x = F.sigmoid(x)
        # print('output sigmoid layer: ', x.shape)
        return x

### initialize and send the model to the device above ### 
model = BassetCNN().to(device)

### define hyperparameters as described in the paper ###
learning_rate = 0.001 # my own arbitrary choice for the lr 
criterion = nn.BCELoss() # this is taken from the paper 
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

### define some other parameters up here for use downstream in training/testing ### 
num_epochs = 50
n_batches = 6 # re-defined here purely for the purpose of training

### train the model ### 
### i didn't include any measures of accuracy of anything here -- just loss -- because i didn't know what the ground truth data should really look like? ### 
def train_model():
    ### initialize the weights. unsure if they did random initialization however ### 
    def init_weights(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        model.apply(init_weights)
 
    train_log = {'training_loss_per_epoch':[]} # keep track of params 

    for epoch in range(num_epochs):
        # switch the model to training mode 
        model.train() 
        t_loss_per_batch = []
        for i, (samples, labels) in enumerate(train_dataloader): 
            samples = samples.permute(1, 0) # tranpose the sample so that it is (4x600)
            samples = samples.to(device) # send the samples to the device

            labels = labels[0] # assume that the first row is just the labels (i'm not sure how the real input data will be formatted, but this definitely isn't a great assumption)
            labels = labels.to(device) 
            model.to(device) 
            predicted = model(samples) # get the predicted output 
            loss = criterion(predicted, labels) 
            t_loss_per_batch.append(loss.item()) 

            optimizer.zero_grad() # zero accumulated gradients 
            loss.backward() # backprop
            optimizer.step() # step on params 
            
            # print message for every batch 
            if (i+1) % n_batches == 0: # -- this is where the i term above is used in for loop
                print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{int(n_total_steps)}], Batch Loss: {loss.item():.4f}')
        
        # insert the AVERAGE batch training loss -> log 
        t_loss = sum(t_loss_per_batch) / len(t_loss_per_batch)
        train_log['training_loss_per_epoch'].append(t_loss)

    # save the model's params after fully training it 
    PATH = './basset_params.pth'
    torch.save(model.state_dict(), PATH) # -- save the model's params 

    return train_log # return the log of loss values 

### test the model; load in saved params from specified PATH ### 
def test_model(PATH):
    # load in the state dict from the passed in path 
    model.load_state_dict(torch.load(PATH)) 
    model.eval() # switch to eval mode to switch off layers like dropout 

    ### initialize log dictionary for keeping track of training loss, accuracy, etc.. ### 
    test_log = {'testing_loss_per_epoch':[]}
    
    with torch.no_grad():
        for epoch in range(num_epochs):
            testing_loss_per_batch = []
            for i, (samples, labels) in enumerate(test_dataloader):
                # set testing samples 
                samples = samples.permute(1, 0)
                samples = samples.to(device)
                # set testing labels 
                labels = labels[0]
                labels = labels.to(device)
                # send the model to the device
                model.to(device)
                predicted = model(samples)
                # calculate the loss 
                loss = criterion(predicted, labels)
                testing_loss_per_batch.append(loss.item())
            # calculate the average test loss 
            test_loss = sum(testing_loss_per_batch) / len(testing_loss_per_batch)
            test_log['testing_loss_per_epoch'].append(test_loss)
    return test_log

# call the methods to train and test the model 
train_log = train_model()
test_log = test_model('./basset_params.pth')

### quick function to get a visual of the loss ###
### of note: strange results for testing loss -- constant -- but i still have to debug and see what the specific issue is ### 
def plot_loss(num_epochs, train_loss, test_loss):
    f, a = plt.subplots(figsize=(10,7.5), layout='constrained') # don't need to specify 2x2 or anything here, bc i'm just going to plot the loss 
    f.suptitle('Calculated Loss')
    a.plot(num_epochs, train_loss, label='Training Loss')
    a.plot(num_epochs, test_loss, label=f'Testing Loss')
    a.set_xlabel('Number of Epochs')
    a.set_ylabel('Average Loss')
    a.set_title(f'Training and Testing Loss')
    a.legend()
    plt.show() # just show it - i guess i'll save it later once i have a more functional model running 
 
# call the loss function plotter 
plot_loss(np.linspace(1, num_epochs, num=num_epochs).astype(int), train_log['training_loss_per_epoch'], test_log['testing_loss_per_epoch'])