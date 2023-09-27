from utility import load_dataframes

from models import Classifier_1, Classifier_2, Classifier_3, train, evaluate, test, save_model

from data_transform import ImageDataset

from torch.utils.data import DataLoader, SubsetRandomSampler

from sklearn.model_selection import KFold

import torch.optim as optim
import torch.nn as nn

import torch

from tqdm import tqdm

import numpy as np

import copy


hyperparameters={
    "loss" : nn.CrossEntropyLoss(),
    "batch_size" : 64,
    "patient" : 20,
    "lr" : 0.01,
    "momentum" : 0.5
}

def k_fold_cross_validation(hyperparameters:dict, untrained_model, model_name = "prova", k_folds = 10, max_epoch = 500, device="cuda"):
    
    loss = hyperparameters["loss"]
    batch_size = hyperparameters["batch_size"]
    patient = hyperparameters["patient"]
    
    lr = hyperparameters["lr"]
    momentum = hyperparameters["momentum"]
    
    (train_X, train_y) = load_dataframes(is_train=True)
    train_dataset = ImageDataset(train_X, train_y)

    # Initialize the k-fold cross validation
    kf = KFold(n_splits=k_folds, shuffle=True)

    best_losses=[]

    folds_bar = tqdm(enumerate(kf.split(train_X)), total=k_folds, desc=f'FOLD [0 / {k_folds}]', leave=True)

    for fold, (train_idx, val_idx) in folds_bar:
        
        fold += 1
        
        folds_bar.write("------")
        folds_bar.write(f"FOLD {fold}\n")

        # Define the data loaders for the current fold
        train_loader = DataLoader(
            dataset = train_dataset,
            batch_size = batch_size,
            sampler = SubsetRandomSampler(train_idx.tolist()),
        )
        
        val_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            sampler = SubsetRandomSampler(val_idx.tolist()),
        )
        
        # Initialize the model and optimizer
        model = copy.deepcopy(untrained_model).to(device)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

        # Train the model on the current fold
        epochs_bar = tqdm(range(1, max_epoch), desc=f'TRAIN Patient [0 / {patient}]', leave=False)
        
        best_loss=np.inf
        no_improvements=0
        
        for epoch in epochs_bar:
            
            train_loss = train(model, device, train_loader, optimizer, epoch)
            
            eval_loss = evaluate(model, device, val_loader, epoch)
            
            if eval_loss<best_loss:
                
                #epochs_bar.write(f"decreased loss at epoch {epoch}: SAVING MODEL")
                
                no_improvements=0
                best_loss = eval_loss
                
                best_losses.append(best_loss)
                
                save_model(model, optimizer, train_loss, eval_loss, epoch, fold, model_name)
            else:
                no_improvements += 1
            
            if no_improvements==patient:
                epochs_bar.write("no improvements... early stopping!")
                break
            
            epochs_bar.set_description(f'TRAIN Patient [{no_improvements} / {patient}]')
            epochs_bar.set_postfix(eval_loss = eval_loss, train_loss=train_loss)
        
        #folds_bar.write("\n")
        
        
        (test_X, test_y) = load_dataframes(is_train=False)

        test_dataset = ImageDataset(test_X, test_y)

        test_loader = DataLoader(
            dataset = test_dataset,
            batch_size = batch_size,
        )

        accuracy = test(model,device,test_loader)

        folds_bar.write(f"accuracy: {accuracy}\n")
        
        folds_bar.set_description(f'FOLD [{fold} / {k_folds}]')
        folds_bar.set_postfix(best_loss=best_loss)

    return np.average(best_losses)

k_fold_cross_validation(hyperparameters, Classifier_3(),"prova")
