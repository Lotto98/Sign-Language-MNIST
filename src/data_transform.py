from torch.utils.data import Dataset

import pandas as pd
import numpy as np

import torch

from torch.utils.data import random_split

import torchvision.transforms as T

from typing import Tuple

class ImageDataset(Dataset):
    def __init__(self, dataframe_X:pd.DataFrame, series_Y:pd.Series, transform_dimension:Tuple[int,int], device:str="cuda") -> None:
        super().__init__()
        
        assert transform_dimension[0] > 0 and transform_dimension[1] > 0, "Resize dimensions should be > 0"
        
        assert transform_dimension[0]==transform_dimension[1], "Resize dimensions should be equal"
        
        numpy_X = dataframe_X.to_numpy(dtype=np.float32)
        numpy_X = numpy_X.reshape(len(dataframe_X),1,28,28)
        
        self._X = torch.from_numpy(numpy_X).to(device)
        self._X = T.Resize(transform_dimension, antialias = False)(self._X) # type: ignore
        
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