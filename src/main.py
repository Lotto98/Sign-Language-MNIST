from NeuralNetwork import NeuralNetwork

from models import Classifier_1, Classifier_2, Classifier_3, LeNet5

import torch.optim as optim
import torch.nn as nn

from typing import Tuple

from itertools import product, combinations

from tqdm import tqdm

import os

import pandas as pd

import time

def single_test(model:nn.Module,
                model_name:str,
                batch_size:int,
                patience:int,
                data_augmentation_perc:float,
                lr:float=0.0001, model_input_dim:Tuple[int,int] = (28,28))->Tuple[float,float,float]:
    
    optimizer = optim.Adam(model.parameters(),lr=lr, amsgrad=True)

    net = NeuralNetwork(model, optimizer, model_name, model_input_dim, batch_size, patience, data_augmentation_perc)

    train_time = net.full_training()
    
    start = time.perf_counter()
    accuracy = net.test()
    end = time.perf_counter()
    
    return accuracy, end - start, train_time

def multiple_test(model_name:str, n_conv_layers:int):
    
    lrs = [0.0005, 0.0001, 0.00001, 0.000001]
    batch_sizes = [32, 64, 128, 256, 512]
    patiences = [5, 10, 15, 20]
    layers = ["Conv"+str(i) for i in range(2, n_conv_layers+1)]
    layers += ["FC1"]
    layers_all_combinations = [list(comb) for i in range(1, len(layers)+1) for comb in combinations(layers,i)]
    layers_all_combinations.append([])
    n_neurons_molt_factors = [2/3, 1, 2]
    data_augmentation_percs=[0, 0.25, 0.5, 0.75]
    
    results = {
        "model_name":[],
        "lr":[],
        "batch_size":[],
        "patience":[],
        "do_dropout":[],
        "n_neurons_molt_factor":[],
        "data_augmentation_perc":[],
        "test_accuracies": [],
        "test_times":[],
        "train_times":[],
    }
    
    prod = [x for x in product(lrs, batch_sizes, patiences, layers_all_combinations, n_neurons_molt_factors, data_augmentation_percs)]
    
    bar = tqdm(enumerate(prod), total=len(prod))
    
    for i, (lr, batch_size, patience, do_dropout,  n_neurons_molt_factor, data_augmentation_perc) in bar:
        
        os.system('clear')
        
        bar.set_description(f"Test with parameters: lr={lr}, batch_size={batch_size}, patience={patience}, do_dropout={do_dropout}, n_neurons_molt_factor={n_neurons_molt_factor}, data_augmentation_perc={data_augmentation_perc}")
        
        name = model_name + f"_test_{i}"
        
        match model_name:
            case "Classifier_1":
                model = Classifier_1(n_neurons_molt_factor=n_neurons_molt_factor, do_dropout=do_dropout)
            case "Classifier_2":
                model = Classifier_2(n_neurons_molt_factor=n_neurons_molt_factor, do_dropout=do_dropout)
            case "Classifier_3":
                model = Classifier_3(n_neurons_molt_factor=n_neurons_molt_factor, do_dropout=do_dropout)
            case "LeNet5":
                model = LeNet5()
            case _:
                raise IOError("Model not found")
        
        results["model_name"].append(name)
        results["lr"].append(lr)
        results["batch_size"].append(batch_size)
        results["patience"].append(patience)
        results["do_dropout"].append(do_dropout)
        results["n_neurons_molt_factor"].append(n_neurons_molt_factor)
        results["data_augmentation_perc"].append(data_augmentation_perc)
        
        accuracy, train_time, test_time = single_test(model=model,
                                                        model_name=name,
                                                        lr=lr, batch_size=batch_size,
                                                        patience=patience, data_augmentation_perc=data_augmentation_perc)
        
        results["test_accuracies"].append(accuracy)
        results["test_times"].append(test_time)
        results["train_times"].append(train_time)
        
        pd.DataFrame(results).to_parquet(f"../models/{model_name}_results.parquet")

multiple_test("Classifier_2", 2)
#single_test(Classifier_2(2,["FC1"]),"PROVA",128,20,0.75, 0.0001)