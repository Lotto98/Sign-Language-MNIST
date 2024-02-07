from NeuralNetwork import NeuralNetwork, Model, Classifier_1, Classifier_2, Classifier_3, LeNet5

import torch

from typing import Tuple

from itertools import product, combinations

from tqdm import tqdm

import os

import pandas as pd

import time

def single_test(model_name:str,
                model:Model,
                device:torch.device,
                batch_size:int,
                patience:int,
                data_augmentation_perc:float,
                optimizer:str,
                lr:float=0.0001, model_input_dim:Tuple[int,int] = (28,28))->Tuple[float,float,float]:

    net = NeuralNetwork(model_name, model, device, optimizer,
                        lr, model_input_dim, batch_size, patience, data_augmentation_perc)

    train_time = net.full_training()
    
    start = time.perf_counter()
    accuracy, _ = net.test()
    end = time.perf_counter()
    
    return accuracy, end - start, train_time

def single_architecture_tests(results: dict, model: Model, architecture_name:str, model_input_dim:Tuple[int,int], device:torch.device):
    
    optimizers=["AMSGrad","ADAM"]
    lrs = [0.0005, 0.0001, 0.00001, 0.000001]
    batch_sizes = [32, 64, 128, 256, 512]
    patiences = [5, 10, 15, 20]
    data_augmentation_percs=[0, 0.25, 0.5, 0.75]
    
    results.update({
        "optimizer":[],
        "lr":[],
        "batch_size":[],
        "patience":[],
        "data_augmentation_perc":[],
        "test_accuracies": [],
        "test_times":[],
        "train_times":[],
    })
    
    prod = [x for x in product(optimizers, lrs, batch_sizes, patiences, data_augmentation_percs)]
    
    bar = tqdm(enumerate(prod), total=len(prod), desc = "Parameters")
    
    for i, (optimizer, lr, batch_size, patience, data_augmentation_perc) in bar:
        
        bar.set_description(f"Training parameters: optimizer={optimizer}, lr={lr}, batch_size={batch_size}, patience={patience}, data_augmentation_perc={ data_augmentation_perc}")
        
        name = architecture_name +f"_test_{i}"
        
        results["optimizer"].append(optimizer)
        results["lr"].append(lr)
        results["batch_size"].append(batch_size)
        results["patience"].append(patience)
        results["data_augmentation_perc"].append(data_augmentation_perc)
        
        accuracy, test_time, train_time = single_test(name, model, device,
                                                        batch_size, patience, data_augmentation_perc, optimizer, lr, model_input_dim) 
        
        results["test_accuracies"].append(accuracy)
        results["test_times"].append(test_time)
        results["train_times"].append(train_time)
        
        pd.DataFrame(results).to_parquet(f"../models/{architecture_name}_results.parquet")
        

def multiple_architectures_tests(model_name:str, n_conv_layers:int):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_input_dim = (28,28)
    
    layers = ["Conv"+ str(i) for i in range(2, n_conv_layers+1)]
    layers += ["FC1"]
    layers_all_combinations = [list(comb) for i in range(1, len(layers)+1) for comb in combinations(layers,i)]
    layers_all_combinations.append([])

    n_neurons_molt_factors = [0.6, 1, 2]
    
    prod = list(product(n_neurons_molt_factors, layers_all_combinations))
    
    for i, (n_neurons_molt_factor, do_dropout) in enumerate(prod):
        
        print_architecture_spec=True
        
        results={
            'n_neurons_molt_factor':n_neurons_molt_factor,
            'do_dropout':str(do_dropout)
        }
        
        match model_name:
            case "Classifier_1":
                model = Classifier_1(device, n_neurons_molt_factor=n_neurons_molt_factor, do_dropout=do_dropout)
            case "Classifier_2":
                model = Classifier_2(device, n_neurons_molt_factor=n_neurons_molt_factor, do_dropout=do_dropout)
            case "Classifier_3":
                model = Classifier_3(device, n_neurons_molt_factor=n_neurons_molt_factor, do_dropout=do_dropout)
            case _:
                raise IOError("Model not found")
            
        os.system('clear')
        
        print(f"Architecture {i+1}/{len(prod)}")
            
        architecture_name = model.print_architecture(print_architecture_spec)
        
        print("\n=============================HYPERPARAMETERS TUNING=============================")

        single_architecture_tests(results, model, architecture_name, model_input_dim, device)
        
single_architecture_tests(dict(), LeNet5(torch.device("cuda")), "LeNet5", (32,32), torch.device("cuda"))
#multiple_architectures_tests("Classifier_1", 0)
multiple_architectures_tests("Classifier_2", 2)
multiple_architectures_tests("Classifier_3", 3)