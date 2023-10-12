
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from ImageDataset import ImageDataset
from utility import load_dataframes

import os
from tqdm import tqdm
from typing import Tuple

from sklearn.model_selection import train_test_split

import time
    
class NeuralNetwork():
    def __init__( self, model_name:str,
                    model: nn.Module,
                    device:torch.device,
                    lr:float,
                    model_input_dim:Tuple[int, int] = (28,28),
                    batch_size=128, patience:int=20, data_augmentation_perc:float=0,
                    max_epoch:int = 100):
        
        self.device = device
        
        self.__model_name=model_name
        self.__model = model.to(self.device)
        
        #hyperparameters
        self.__lr = lr
        self.max_epoch = max_epoch
        self.__patience = patience
        self.__batch_size = batch_size
        self.__data_augmentation_perc = data_augmentation_perc
        
        #optimizer
        self.__optimizer = optim.Adam(model.parameters(),lr = lr, amsgrad=True)
        
        #loss
        self.__loss = nn.CrossEntropyLoss()
        
        #training
        self.__current_epoch = 0
        self.__stats ={"epochs":[],
                        "train_losses":[],
                        "eval_losses":[],
                        #"current_lrs":[],
                        "test_accuracies":[],
                        #"best_epoch":-1,
                        "training_time_per_epoch": []}
        
        self.__create_dataloaders(model_input_dim)
        
    def __create_dataloaders(self, model_input_dim:Tuple[int, int]):
        
        #training/validation datasets
        train_X, train_y = load_dataframes(isTrain=True)
        
        train_X, val_X, train_y, val_Y = train_test_split(train_X, train_y, test_size=0.2, random_state=8)
        
        train_dataset = ImageDataset(train_X, train_y,
                                        transform_dimension=model_input_dim, data_augmentation_perc = self.__data_augmentation_perc, 
                                        device=self.device)
        val_dataset = ImageDataset(val_X, val_Y,
                                    transform_dimension=model_input_dim, data_augmentation_perc=0, device=self.device)

        # Define the data loaders
        self.__train_loader = DataLoader(
            dataset = train_dataset,
            batch_size = self.__batch_size,
        )

        self.__val_loader = DataLoader(
            dataset = val_dataset,
            batch_size = self.__batch_size,
        )
        
        #test dataset loading
        (test_X, test_y) = load_dataframes(isTrain=False)

        test_dataset = ImageDataset(test_X, test_y,
                                    transform_dimension=model_input_dim,
                                    data_augmentation_perc=0, device=self.device)
        
        self.__test_loader = DataLoader(
            dataset = test_dataset,
            batch_size = self.__batch_size,
        )
        
    def get_current_epoch(self):
        return self.__current_epoch

    def __train(self):
        self.__model.train()
        
        train_loss=0
        
        for data, target in self.__train_loader:
            
            output = self.__model(data.to(self.device))
            
            l = self.__loss(output, target)
            
            train_loss += l.item()
            
            self.__optimizer.zero_grad()
            l.backward()
            
            self.__optimizer.step()
            
        train_loss /= len(self.__train_loader.dataset) # type: ignore
        
        return train_loss

    def __evaluate(self):
        
        self.__model.eval()
        
        val_loss = 0
        with torch.no_grad():
            for data, target in self.__val_loader:
                
                output = self.__model(data.to(self.device))
            
                l = self.__loss(output, target)
                
                val_loss += l.item()

            val_loss /= len(self.__val_loader.dataset) # type: ignore
        
            return val_loss

    def test(self):
        
        assert self.__current_epoch != 0, "Train the model first before testing it"
        
        self.__model.eval()
        
        accuracy = 0
        with torch.inference_mode():
            for data, target in self.__test_loader:
                
                output = self.__model(data.to(self.device))
            
                output_prob = torch.softmax(output, dim=1)
                prediction = torch.argmax(output_prob, dim=1)
                
                accuracy += torch.sum(prediction==target).item()
                
                
            accuracy = accuracy / len(self.__test_loader.dataset) # type: ignore
        
        return accuracy
    
    def __save_model(self):
        
        model_path = os.getcwd()+"/../models/"+self.__model_name+"/"
        if not os.path.exists(model_path):
            os.mkdir(model_path) 
        
        torch.save({
            'epoch': self.__current_epoch,
            'stats': self.__stats,
            'model_state_dict': self.__model.state_dict(),
            'optimizer_state_dict': self.__optimizer.state_dict(),
            }, model_path+"best.tar")
    
    @staticmethod
    def __load(model_name:str):
        model_path = os.getcwd()+"/../models/"+model_name+"/"
        if not os.path.exists(model_path):
            os.mkdir(model_path) 
            
        model_path += "best.tar"
        
        assert os.path.exists(model_path), "No model found" 
        
        return torch.load(model_path)

    def load_model(self): 
        
        checkpoint = NeuralNetwork.__load(self.__model_name)
        
        self.__model.load_state_dict(checkpoint.pop('model_state_dict'))
        self.__model.to(self.device)
        
        self.__optimizer.load_state_dict(checkpoint.pop('optimizer_state_dict'))
        
        self.__current_epoch = checkpoint["epoch"]
        self.__stats = checkpoint["stats"]
        
    @staticmethod
    def return_stats(model_name:str):
        return NeuralNetwork.__load(model_name)["stats"]
    class __EarlyStopper:
        def __init__(self, patience=20):
            self.patience = patience
            self.counter = 0
            self.min_validation_loss = float('inf')
            
            self.save_model = True
            self.stop = False

        def __call__(self, validation_loss):
            
            self.difference = validation_loss - self.min_validation_loss
            
            if self.difference < 0:
                self.min_validation_loss = validation_loss
                self.counter = 0
                self.save_model = True
            else:
                self.counter += 1
                self.save_model = False
                if self.counter >= self.patience:
                    self.stop = True
                    
    
    def full_training(self)->float:

        epochs_bar = tqdm(range(self.__current_epoch + 1, self.max_epoch+1), desc=f'Patient [0 / {self.__patience}]', leave=False)
            
        early_stopper = self.__EarlyStopper(patience=self.__patience)
        
        for epoch in epochs_bar:
            
            self.__current_epoch = epoch
            
            start_training_time = time.perf_counter()
            
            train_loss = self.__train()
            eval_loss = self.__evaluate()
            
            early_stopper(eval_loss)
            
            end_training_time = time.perf_counter()
            
            accuracy = self.test()
            
            self.__stats["epochs"].append(epoch)
            self.__stats["train_losses"].append(train_loss)
            self.__stats["eval_losses"].append(eval_loss)
            self.__stats["training_time_per_epoch"].append( end_training_time - start_training_time )
            self.__stats["test_accuracies"].append(accuracy)
            
            epochs_bar.set_description(f'Patient [{early_stopper.counter} / {self.__patience}]')
            epochs_bar.set_postfix(train_loss=train_loss, eval_loss = eval_loss, difference=early_stopper.difference, test_accuracy=accuracy)
            
            if early_stopper.save_model:
                self.__save_model()
    
            if early_stopper.stop:
                epochs_bar.write(f"Early stopped at epoch {epoch}")
                break
        
        self.load_model() #load best model configuration
        
        return sum(self.__stats["training_time_per_epoch"])
    
    def prova(self):
        if self.__current_epoch != 0:
            assert self.__current_epoch < self.max_epoch, f"Model already trained for {self.max_epoch} epochs: change this value to continue training"
            print(f"Restarting training from epoch {self.__current_epoch + 1}")