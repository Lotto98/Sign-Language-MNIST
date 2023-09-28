import torch.nn as nn
import torch.optim as optim

from utility import load_dataframes

from data_transform import ImageDataset

from torch.utils.data import DataLoader, random_split

import numpy as np
import torch

import os

loss = nn.CrossEntropyLoss()

class Classifier_3(nn.Module):
    def __init__(self, n_neurons = 64):
        super(Classifier_3, self).__init__()
        
        self.Conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 5), #24x24x32
            nn.MaxPool2d(2), #12x12x32
            nn.ReLU()
        )
        self.Conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5), #8x8x64
            nn.MaxPool2d(2), #4x4x64
            nn.ReLU()
        )
        self.Conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3), #2x2x128
            nn.MaxPool2d(2), #1x1x128
            nn.ReLU()
        )
        
        self.Linear1 = nn.Linear(128, n_neurons)
        self.Linear2 = nn.Linear(n_neurons, 32)
        self.Linear3 = nn.Linear(32, 25)
        
    def forward(self, x):
        x = self.Conv1(x)
        x = self.Conv2(x)
        x = self.Conv3(x)
        x = x.view(x.size(0), -1) #flatten
        x = self.Linear1(x)
        x = self.Linear2(x)
        x = self.Linear3(x)
        return x

class Classifier_2(nn.Module):
    def __init__(self):
        super(Classifier_2, self).__init__()
        self.Conv1 = nn.Sequential(
        nn.Conv2d(1, 16, 5), # 24x24x16 
        nn.MaxPool2d(2), # 12x12x16
        nn.ReLU()
        )
        
        self.Conv2 = nn.Sequential(
        nn.Conv2d(16, 32, 5), # 8x8x32
        nn.MaxPool2d(2),  # 4x4x32
        nn.ReLU()
        )
        
        self.Linear1 = nn.Linear(32 * 4 * 4, 64)
        self.Linear2 = nn.Linear(64, 32)
        self.Linear3 = nn.Linear(32, 25)
        
    def forward(self, x):
        x = self.Conv1(x)
        x = self.Conv2(x)
        x = x.view(x.size(0), -1) #flatten
        x = self.Linear1(x)
        x = self.Linear2(x)
        x = self.Linear3(x)
        return x
    
class Classifier_1(nn.Module):
    def __init__(self):
        super(Classifier_1, self).__init__()
        self.Conv1 = nn.Sequential(
        nn.Conv2d(1, 8, 5), # 24x24x8 
        nn.MaxPool2d(2), # 12x12x8
        nn.ReLU()
        )
        
        self.Linear1 = nn.Linear(8 * 12 * 12, 64)
        self.Linear2 = nn.Linear(64, 32)
        self.Linear3 = nn.Linear(32, 25)
        
    def forward(self, x):
        x = self.Conv1(x)
        x = x.view(x.size(0), -1) #flatten
        x = self.Linear1(x)
        x = self.Linear2(x)
        x = self.Linear3(x)
        return x

# Define the training function
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    
    train_loss=0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        
        output = model(data)
        
        l = loss(output, target)
        
        train_loss += l.item()
        
        #print(data.shape, output.shape, target.shape)
        
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        
    train_loss /= len(train_loader.dataset)
    
    return train_loss

# Define the evaluation function
def evaluate(model, device, val_loader, epoch):
    model.eval()
    
    val_loss = 0
    with torch.no_grad():
        for data, target in val_loader:
            
            output = model(data)
        
            l = loss(output, target)
            
            val_loss += l.item()

        val_loss /= len(val_loader.dataset)
    
        return val_loss

def test(model, device, test_loader):
    model.eval()
    
    accuracy = 0
    test_loss = 0
    with torch.inference_mode():
        for data, target in test_loader:
            
            output = model(data)
        
            output_prob = torch.softmax(output, dim=1)
            prediction = torch.argmax(output_prob, dim=1)
            
            accuracy += torch.sum(prediction==target).item()
            
            l = loss(output, target)
            
            test_loss += l.item()
            
        accuracy = accuracy/len(test_loader.dataset)
        test_loss /= len(test_loader.dataset)
    
    
    return accuracy, test_loss

def save_model(model, optimizer, training_loss_value, validation_loss_value, epoch, fold, model_name):
    
    folder = "best_models"
    
    model_path = os.getcwd()+"/../models/"+model_name+"/"
    if not os.path.exists(model_path):
        os.mkdir(model_path) 
    
    model_path += folder+"/"
    if not os.path.exists(model_path):
        os.mkdir(model_path) 
    
    torch.save({
        'epoch': epoch,
        'loss': training_loss_value,
        'validation' : validation_loss_value,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
        }, model_path+"fold_"+str(fold)+".tar")

def spit_train(train_data, perc_val_size):
    
    train_size = len(train_data)
    val_size = int((train_size * perc_val_size) // 100)
    train_size -= val_size

    return random_split(train_data, [int(train_size), int(val_size)]) #train_data, val_data 

def full_training():
    # Define the device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 64

    (train_X, train_y) = load_dataframes(is_train=True)
    train_dataset = ImageDataset(train_X, train_y)

    train_dataset, val_dataset = spit_train(train_dataset,20)

    # Define the data loaders
    train_loader = DataLoader(
        dataset = train_dataset,
        batch_size = batch_size,
    )

    val_loader = DataLoader(
        dataset = val_dataset,
        batch_size=batch_size,
    )

    # Initialize the model and optimizer
    model = Classifier().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    # Train the model on the current fold
    for epoch in range(1, 150):
        train_loss = train(model, device, train_loader, optimizer, epoch)
        val_loss = evaluate(model, device, val_loader, epoch)
        print(epoch, train_loss, val_loss)


    (test_X, test_y) = load_dataframes(is_train=False)

    test_dataset = ImageDataset(test_X, test_y)

    test_loader = DataLoader(
        dataset = test_dataset,
        batch_size = batch_size,
    )

    accuracy = test(model,device,test_loader)

    print(accuracy)

#full_training()