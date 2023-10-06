import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import gc

import os

from typing import Union, Tuple

from sklearn.model_selection import train_test_split

response_transform={n:chr(n+65) for n in range(0,26)}

def get_dataset(path:str):
    
    data=pd.read_csv(path)
    data_X=data.drop('label',axis=1)
    data_Y=data['label']
    
    del data
    gc.collect()
    
    return data_X,data_Y

def plot_images(train_X:pd.DataFrame, train_Y:pd.Series, ran:int=666):
    
    fig, axs = plt.subplots(5,6,figsize=(10,10))

    axs = [item for sublist in axs for item in sublist]

    for k,ax in enumerate(axs):
        ax.axis("off")
        
        if k<=25:
            ax.set_title(response_transform[k])
            
            if k!=9 and k!=25: #impossible to distinguish J and Z
                
                indexes = train_Y[train_Y==k].index
                i = indexes[ran % len(indexes)]
                
                to_show=np.reshape(train_X.iloc[i].to_numpy(),(28,28)) # type: ignore
            else:
                to_show=np.zeros((28,28))
            
            ax.imshow(to_show,cmap='gray')
            
def save_dataframes(train_X:pd.DataFrame, test_X:pd.DataFrame,
                    train_Y:pd.Series, test_Y:pd.Series):
    
    dfs_path = os.getcwd()+"/../data/dataframes" 
    if not os.path.exists(dfs_path):
        os.makedirs(dfs_path)

    train_X.to_parquet(dfs_path+"/train_X.parquet")
    train_Y.to_frame().to_parquet(dfs_path+"/train_Y.parquet")

    test_X.to_parquet(dfs_path+"/test_X.parquet")
    test_Y.to_frame().to_parquet(dfs_path+"/test_Y.parquet")
    
    
def load_dataframes(isTrain:bool):#->Union[Tuple[pd.DataFrame,pd.Series, pd.DataFrame,pd.Series], Tuple[pd.DataFrame,pd.Series]]:
    
    dfs_path = os.getcwd()+"/../data/dataframes" 
    if not os.path.exists(dfs_path):
        raise FileNotFoundError("dataframes directory does not exist. Please execute dataset notebook first.")
    
    if isTrain:
        dfs_path += "/train_"
    else:
        dfs_path += "/test_"
        
    X = pd.read_parquet(dfs_path+"X.parquet")
    Y = pd.read_parquet(dfs_path+"Y.parquet").squeeze()

    #print(type(X),type(Y))
    
    if isTrain:
        X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
        
        #print((X_train), type(X_val), type(y_train), type(y_val))
        
        return X_train, X_val, y_train, y_val
    else:
        return X, Y