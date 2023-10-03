from NeuralNetwork import NeuralNetwork

from models import Classifier_3, LeNet5

import torch.optim as optim
import torch.nn as nn

from typing import Tuple

from itertools import product

from tqdm import tqdm

import os

def single_test(model:nn.Module, model_name:str, batch_size:int, patience:int, lr:float=0.0001, model_input_dim:Tuple[int,int] = (28,28)):
    
    optimizer = optim.Adam(model.parameters(),lr=lr, amsgrad=True)

    net = NeuralNetwork(model, optimizer, model_name, model_input_dim, batch_size, patience)

    net.full_training()

def multiple_test(model:nn.Module, model_name:str):
    
    lrs = [0.1, 0.001, 0.0001]
    batch_sizes = [16, 32, 64, 128, 256]
    patiences = [5, 10, 15, 20]
    
    prod=[x for x in product(lrs, batch_sizes, patiences)]
    
    bar = tqdm(enumerate(prod), total=len(prod))
    
    for i,(lr, batch_size, patience) in bar:
        
        os.system('clear')
        
        bar.set_description(f"Test with parameters: lr={lr}, batch_size={batch_size}, patience={patience}")
        
        name = model_name + f"_test_{i}"
        
        single_test(model=model, model_name=name, lr=lr, batch_size=batch_size, patience=patience )
    
multiple_test(model=Classifier_3(), model_name="Classifier_3")