
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from ImageDataset import ImageDataset, T
from utility import load_dataframes, response_transform, plot_image

import os
from tqdm import tqdm
from typing import Tuple, Dict

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display  import display

from base_models import Model, Classifier_1, Classifier_2, Classifier_3, LeNet5

class NeuralNetwork():
    
    @staticmethod
    def load_NN(model_hyperparameters:pd.Series, architecture_id_to_model_name: dict, device: torch.device, model_input_dim: Tuple[int, int]):
        
        model_name = architecture_id_to_model_name[model_hyperparameters["architecture_id"]] + "_test_" + str(int(model_hyperparameters["test_id"]))
        
        print_spec=True
        
        if "Classifier_1" in model_name:
            model = Classifier_1(device=device,
                                    n_neurons_molt_factor=float(model_hyperparameters["n_neurons_molt_factor"]),
                                    do_dropout=model_hyperparameters["do_dropout"])
        elif "Classifier_2" in model_name:
            model = Classifier_2(device=device,
                                    n_neurons_molt_factor=float(model_hyperparameters["n_neurons_molt_factor"]),
                                    do_dropout=model_hyperparameters["do_dropout"])
        elif "Classifier_3" in model_name:
            model = Classifier_3(device=device,
                                    n_neurons_molt_factor=float(model_hyperparameters["n_neurons_molt_factor"]),
                                    do_dropout=model_hyperparameters["do_dropout"])
        elif "LeNet5" in model_name:
            model = LeNet5(device=device)
            print_spec = False
        else:
            raise IOError("Model not found")
        
        neuralNetwork = NeuralNetwork(model_name, model, device, model_hyperparameters["optimizer"], float(model_hyperparameters["lr"]),
                                model_input_dim, int(model_hyperparameters["batch_size"]),
                                int(model_hyperparameters["patience"]), float(model_hyperparameters["data_augmentation_perc"]))
        
        neuralNetwork.__load_model()
        
        print(f"Loaded model: {model_name}")
        print("\n")
        
        print("Architecture:")
        neuralNetwork.print_architecture(print_spec)
        
        return neuralNetwork
        
    def print_architecture(self,print_architecture_spec:bool):
        self.__model.print_architecture(print_architecture_spec)
    
    def __init__( self, model_name:str,
                    model: Model,
                    device: torch.device,
                    optimizer: str,
                    lr:float=0.001,
                    model_input_dim:Tuple[int, int] = (28,28),
                    batch_size=128, patience:int=20, data_augmentation_perc:float=0,
                    max_epoch:int = 100):
        
        self.device = device
        
        self.__model_name=model_name
        self.__model = model.to(self.device)
        
        #hyperparameters
        self.max_epoch = max_epoch
        self.__patience = patience
        self.__batch_size = batch_size
        self.__data_augmentation_perc = data_augmentation_perc
        
        #optimizer
        match optimizer:
            case "ADAM":
                self.__optimizer = optim.Adam(model.parameters(), lr = lr)
            case "AMSGrad":
                self.__optimizer = optim.Adam(model.parameters(), lr = lr, amsgrad=True)
            case _:
                raise ValueError("Optimizer must be 'ADAM' or 'ASMGrad'")
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
        
        self.__model_input_dim = model_input_dim
        
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
        (self.test_X, self.test_y) = load_dataframes(isTrain=False)

        test_dataset = ImageDataset(self.test_X, self.test_y,
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
        
        predictions=[]
        accuracy = 0
        with torch.inference_mode():
            for data, target in self.__test_loader:
                
                output = self.__model(data.to(self.device))
            
                output_prob = torch.softmax(output, dim=1)
                prediction = torch.argmax(output_prob, dim=1)
                
                predictions.append(prediction.cpu().numpy())
                
                accuracy += torch.sum(prediction==target).item()
                
            accuracy = accuracy / len(self.__test_loader.dataset) # type: ignore
            
        predictions = np.concatenate( predictions, axis=0 )
        
        return accuracy, predictions
    
    def get_wrong_predictions(self):
        
        test_y = self.test_y.to_numpy()
        _, pred_y = self.test()
        
        wrong_predictions = np.where(np.not_equal(test_y, pred_y))[0]
        
        return self.test_X.iloc[wrong_predictions], self.test_y.iloc[wrong_predictions].apply(lambda x:response_transform[x])
    
    def predict(self, image:pd.Series)->Dict[str, float]:
        
        assert self.__current_epoch != 0, "Train the model first before testing it"
        
        numpy_image = image.to_numpy(dtype=np.float32)
        
        torch_image = torch.from_numpy(numpy_image).to(self.device)
        torch_image = torch_image.reshape(1,1,28,28)
        
        torch_image = T.Resize(self.__model_input_dim, antialias = False)(torch_image)  # type: ignore
        
        self.__model.eval()
        with torch.inference_mode():
            output = self.__model(torch_image)
    
        softmax = {response_transform[i]: round(prob, 4) 
                    for i, prob in enumerate(torch.softmax(output, dim=1).cpu().numpy()[0])}
        
        return softmax
    
    def __save_model(self, only_stats:bool=False):
        
        model_path = os.getcwd()+"/../models/"+self.__model_name+"/"
        if not os.path.exists(model_path):
            os.mkdir(model_path) 
        
        if not only_stats:
            torch.save({
                'epoch': self.__current_epoch,
                'stats': self.__stats,
                'model_state_dict': self.__model.state_dict(),
                'optimizer_state_dict': self.__optimizer.state_dict(),
                }, model_path+"best.tar")
        else:
            checkpoint = NeuralNetwork.__load(self.__model_name)
            checkpoint["stats"] = self.__stats
            torch.save(checkpoint, model_path+"best.tar")
    
    @staticmethod
    def __load(model_name:str):
        model_path = os.getcwd()+"/../models/"+model_name+"/"
        if not os.path.exists(model_path):
            os.mkdir(model_path) 
            
        model_path += "best.tar"
        
        assert os.path.exists(model_path), "No model found" 
        
        return torch.load(model_path)

    def __load_model(self): 
        
        checkpoint = NeuralNetwork.__load(self.__model_name)
        
        self.__model.load_state_dict(checkpoint.pop('model_state_dict'))
        self.__model.to(self.device)
        
        self.__optimizer.load_state_dict(checkpoint.pop('optimizer_state_dict'))
        
        self.__current_epoch = checkpoint["epoch"]
        self.__stats = checkpoint["stats"]
        
    
    def return_stats(self):
        return pd.DataFrame(self.__stats)
    
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
        
        to_write = "No early stopping"

        early_stopper = self.__EarlyStopper(patience=self.__patience)
        
        for epoch in epochs_bar:
            
            self.__current_epoch = epoch
            
            start_training_time = time.perf_counter()
            
            train_loss = self.__train()
            eval_loss = self.__evaluate()
            
            early_stopper(eval_loss)
            
            end_training_time = time.perf_counter()
            
            accuracy,_ = self.test()
            
            self.__stats["epochs"].append(epoch)
            self.__stats["train_losses"].append(train_loss)
            self.__stats["eval_losses"].append(eval_loss)
            self.__stats["training_time_per_epoch"].append( end_training_time - start_training_time )
            self.__stats["test_accuracies"].append(accuracy)
            
            epochs_bar.set_description(f'Patient [{early_stopper.counter} / {self.__patience}]')
            epochs_bar.set_postfix(train_loss=train_loss, eval_loss = eval_loss, difference=early_stopper.difference, test_accuracy=accuracy)
            
            if early_stopper.save_model:
                self.__save_model()
            else:
                self.__save_model(only_stats=True)
    
            if early_stopper.stop:
                to_write = f"Early stopped at epoch {epoch}"
                break
            
        #epochs_bar.write(to_write)
        epochs_bar.close()
        
        self.__load_model() #load best model configuration
        
        return sum(self.__stats["training_time_per_epoch"])
            
    def plot_confusion_matrix(self):
        
        fig, ax = plt.subplots(figsize=(20,10))
        ax.set_title(self.__model_name)
        
        accuracy, prediction = self.test()
        
        cm = confusion_matrix(self.test_y, prediction, labels=list(response_transform.keys()))
        ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=list(response_transform.values())).plot(ax=ax)
        plt.show()
        
        print(accuracy)
        
    def explore_wrong_predictions(self):
        
        self.plot_confusion_matrix()
        
        wrong_predictions_images, true_responses = self.get_wrong_predictions()
        
        print("WRONG PREDICTIONS:")
        
        for i, (_, wrong_pred) in enumerate(wrong_predictions_images.iterrows()):
            
            softmax = self.predict(wrong_pred)
            prediction = max(softmax, key=softmax.get) # type: ignore
            
            plot_image(wrong_pred)
            print(f"true response: {true_responses.iloc[i]}")
            print(f"predicted: {prediction}") 
            display(softmax)
            print("=====================================\n\n")
            
    def plot_metrics(self):
        
        fig,ax=plt.subplots(figsize=(20,10))
        
        log_transform = lambda x: np.log([np.finfo(np.float64).tiny if loss==0 else loss for loss in x])
        
        ax.plot(self.__stats["epochs"],(self.__stats["eval_losses"]), color="blue", label="evaluation loss")
        ax.plot(self.__stats["epochs"],(self.__stats["train_losses"]), color="orange", label="training loss")
        ax.axvline(self.__current_epoch, label="best epoch (early stopping)", linestyle="--", color="red")
        ax.legend(loc="upper right", fontsize=14)
        ax.set_title("Training and evaluation loss", fontsize=18)
        ax.set_xlabel("epochs", fontsize=16)
        ax.set_ylabel("loss", fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.tick_params(axis='both', which='minor', labelsize=11)
        
        fig,ax=plt.subplots(figsize=(20,10))
        
        ax.plot(self.__stats["epochs"],log_transform(self.__stats["eval_losses"]), color="blue", label="evaluation loss")
        ax.plot(self.__stats["epochs"],log_transform(self.__stats["train_losses"]), color="orange", label="training loss")
        ax.axvline(self.__current_epoch, label="best epoch (early stopping)", linestyle="--", color="red")
        ax.legend(loc="upper right")
        ax.set_title("Training and evaluation loss (log scale)", fontsize=18)
        ax.set_xlabel("epochs", fontsize=16)
        ax.set_ylabel("loss (log scale)", fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.tick_params(axis='both', which='minor', labelsize=11)

        fig,ax=plt.subplots(figsize=(20,10))
        ax.plot(self.__stats["epochs"],self.__stats["test_accuracies"], label="test accuracy")
        ax.axvline(self.__current_epoch, label="best epoch (early stopping)", linestyle="--", color="red")
        ax.legend(loc="upper right")
        ax.set_title("Test accuracy for each training epoch", fontsize=18)
        ax.set_xlabel("epochs", fontsize=16)
        ax.set_ylabel("accuracy", fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.tick_params(axis='both', which='minor', labelsize=11)
"""
    def custom_images_test(self):
        
        images, responses = load_all_my_images()
        
        accuracy=0
        
        predictions=[]
        
        for image, response in zip(images, responses):
            softmax = self.predict(image)
            
            prediction = max(softmax, key= lambda x: softmax[x])
            predictions.append(prediction)
            
            if response==prediction:
                accuracy+=1
            
            plot_image(image)
            print(f"True response: {response}")
            
            print(f"Predicted response: {prediction}")
            display(self.predict(image))
            print("================================")
        
        print(accuracy/len(images))
        
        cm=confusion_matrix(responses, predictions)
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[x for x in response_transform.values() if x!="Z" and x!="J"]).plot()
"""