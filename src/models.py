import torch.nn as nn
import torch.optim as optim

from utility import load_dataframes

from data_transform import ImageDataset

from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
import torch

import os

from tqdm import tqdm

class LeNet5(nn.Module):
    def __init__(self, num_classes=25):
        super(LeNet5, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        
        self.fc = nn.Linear(400, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out
    
class Classifier_3(nn.Module):
    def __init__(self, n_neurons = 512):
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
        
        self.dropout=nn.Dropout()
        
        self.Linear1 = nn.Linear(128, n_neurons)
        self.relu = nn.ReLU()
        self.Linear2 = nn.Linear(n_neurons, 32)
        self.relu1 = nn.ReLU()
        self.Linear3 = nn.Linear(32, 25)
        
    def forward(self, x):
        x = self.Conv1(x)
        x = self.Conv2(x)
        x = self.Conv3(x)
        x = x.view(x.size(0), -1) #flatten
        x = self.dropout(x)
        x = self.Linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.Linear2(x)
        x = self.relu(x)
        x = self.Linear3(x)
        return x

class Classifier_2(nn.Module):
    def __init__(self, n_neurons = 512):
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
        
        self.Linear1 = nn.Linear(32 * 4 * 4, n_neurons)
        self.Linear2 = nn.Linear(n_neurons, 32)
        self.Linear3 = nn.Linear(32, 25)
        
    def forward(self, x):
        x = self.Conv1(x)
        x = self.Conv2(x)
        x = x.view(x.size(0), -1) #flatten
        x = self.Linear1(x)
        x = self.Linear2(x)
        x = self.Linear3(x)
        return x
    

class NeuralNetwork():
    def __init__( self, model: nn.Module, optimizer, model_name:str, batch_size=128, max_epoch:int=1000,
                 patience:int=20):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model_name=model_name
        
        self.model = model.to(self.device)
        self.optimizer = optimizer
        
        self.loss = nn.CrossEntropyLoss()
        
        self.max_epoch=max_epoch
        self.patience=patience
    
        (train_X, train_y) = load_dataframes(is_train=True)
        train_dataset = ImageDataset(train_X, train_y)

        train_dataset, val_dataset = train_dataset.spit_train_val(0.2)

        # Define the data loaders
        self.train_loader = DataLoader(
            dataset = train_dataset,
            batch_size = batch_size,
        )

        self.val_loader = DataLoader(
            dataset = val_dataset,
            batch_size = batch_size,
        )
        
        (test_X, test_y) = load_dataframes(is_train=False)

        test_dataset = ImageDataset(test_X, test_y)
        
        self.test_loader = DataLoader(
            dataset = test_dataset,
            batch_size = batch_size,
        )

    def train(self):
        self.model.train()
        
        train_loss=0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            
            output = self.model(data.to(self.device))
            
            l = self.loss(output, target)
            
            train_loss += l.item()
            
            self.optimizer.zero_grad()
            l.backward()
            
            self.optimizer.step()
            
        train_loss /= len(self.train_loader.dataset) # type: ignore
        
        return train_loss

    def evaluate(self):
        
        self.model.eval()
        
        val_loss = 0
        with torch.no_grad():
            for data, target in self.val_loader:
                
                output = self.model(data.to(self.device))
            
                l = self.loss(output, target)
                
                val_loss += l.item()

            val_loss /= len(self.val_loader.dataset) # type: ignore
        
            return val_loss

    def test(self):
        self.model.eval()
        
        accuracy = 0
        with torch.inference_mode():
            for data, target in self.test_loader:
                
                output = self.model(data.to(self.device))
            
                output_prob = torch.softmax(output, dim=1)
                prediction = torch.argmax(output_prob, dim=1)
                
                accuracy += torch.sum(prediction==target).item()
                
                
            accuracy = accuracy/len(self.test_loader.dataset) # type: ignore
        
        return accuracy
    
    def __save_model(self, epoch:int, stats:dict):
        
        model_path = os.getcwd()+"/../models/"+self.model_name+"/"
        if not os.path.exists(model_path):
            os.mkdir(model_path) 
        
        torch.save({
            'epoch': epoch,
            'stats':stats,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }, model_path+"best.tar")
        
    def __load_model(self):
        
        model_path = os.getcwd()+"/../models/"+self.model_name+"/"
        if not os.path.exists(model_path):
            os.mkdir(model_path) 
            
        model_path += "best.tar"
        
        checkpoint = torch.load(model_path)
        
        self.model.load_state_dict(checkpoint.pop('model_state_dict'))
        self.model.to(self.device)
        
        self.optimizer.load_state_dict(checkpoint.pop('optimizer_state_dict'))
        
        return checkpoint["epoch"], checkpoint["stats"]

    class EarlyStopper:
        def __init__(self, patience=20):
            self.patience = patience
            self.counter = 0
            self.min_validation_loss = float('inf')
            
            self.save_model=True
            self.stop=False

        def __call__(self, validation_loss):
            
            self.difference = validation_loss - self.min_validation_loss
            
            if self.difference < 0:
                self.min_validation_loss = validation_loss
                self.counter = 0
                self.save_model=True
            else:
                self.counter += 1
                self.save_model=False
                if self.counter >= self.patience:
                    self.stop = True
                    
    
    def full_training(self):
        
        self.trained=True
        
        stats={"epochs":[],
               "train_losses":[],
               "eval_losses":[],
               "current_lrs":[],
               "test_accuracies":[],
               "best_epoch":-1}

        epochs_bar = tqdm(range(1, self.max_epoch+1), desc=f'Patient [0 / {self.patience}]', leave=True)
            
        early_stopper = self.EarlyStopper(patience=self.patience)

        for epoch in epochs_bar:
            
            train_loss = self.train()
            
            eval_loss = self.evaluate()
            
            current_lr = self.optimizer.param_groups[0]["lr"]
            
            early_stopper(eval_loss)
            
            accuracy = self.test()
            
            epochs_bar.set_description(f'Patient [{early_stopper.counter} / {self.patience}]')
            epochs_bar.set_postfix(train_loss=train_loss, eval_loss = eval_loss, lr=current_lr, difference=early_stopper.difference, test_accuracy=accuracy)
            
            stats["epochs"].append(epoch)
            stats["train_losses"].append(train_loss)
            stats["eval_losses"].append(eval_loss)
            stats["current_lrs"].append(current_lr)
            stats["test_accuracies"].append(accuracy)            
            
            if early_stopper.save_model:
                self.__save_model(epoch, stats)
                stats["best_epoch"]=epoch
            
            if early_stopper.stop:
                epochs_bar.write(f"Early stopped at epoch {epoch}")
                break
        
        self.__load_model()
                
        return pd.DataFrame(stats)

model = Classifier_3()
from torchsummary import summary
summary(model, (1,28,28))
optimizer = optim.Adam(model.parameters(),lr=0.0001, amsgrad=True)
net = NeuralNetwork(model,optimizer,"Classifier_3")

stats = net.full_training()

stats.to_parquet(os.getcwd()+"/../models/Classifier_3/stats.parquet")