from torch.utils.data import Dataset

import pandas as pd
import numpy as np

import torch

from torch.utils.data import random_split

class ImageDataset(Dataset):
    def __init__(self, dataframe_X:pd.DataFrame, series_Y:pd.Series, device:str="cuda", transform:bool = False) -> None:
        
        super().__init__()
        
        self._X = torch.from_numpy(dataframe_X.to_numpy(dtype=np.float32).reshape(len(dataframe_X),1,28,28)).to(device)
        self._Y = torch.from_numpy(series_Y.to_numpy()).to(device)
        
    def __len__(self) -> int:
        return len(self._X)
    
    def __getitem__(self, index):
        return self._X[index] , self._Y[index]
    
    def spit_train_val(self, perc_val_size:float):
        
        if perc_val_size < 0 or perc_val_size > 1:
            raise ValueError("Validation percentage should be between 0 and 1")
        
        train_size = len(self)
        val_size = int(train_size * perc_val_size)
        train_size -= val_size

        return random_split(self, [int(train_size), int(val_size)]) #train_data, val_data 