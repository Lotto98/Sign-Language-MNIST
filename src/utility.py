import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import gc

import os

from typing import Tuple

from os import listdir
from os.path import isfile

from random import randint

from PIL import Image


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

def sample_image(index:int=-1):
    
    test_X, test_y  = load_dataframes(isTrain=False)
    
    if index==-1:
        index = randint(0, len(test_X)-1)
        print(f"Sampled image with index {index}")
    
    image = test_X.iloc[10]
    response = test_y.iloc[10]
    
    plot_image(image)
    print(f"True label: {response_transform[response]}")
    
    return image, response

def load_my_image(title:str):
    
    img_path = os.getcwd()+'/../other_data/'+title

    img = Image.open(img_path).convert('L')
    img = img.resize((28,28))

    numpydata = np.asarray(img).reshape(28*28)

    image = pd.Series(numpydata)
    
    return image

def load_all_my_images():
    
    images=[]
    responses=[]
    
    titles = os.listdir(os.getcwd()+'/../other_data/')
    
    for title in titles:
        if isfile(os.getcwd()+'/../other_data/'+title):
            images.append(load_my_image(title))
            responses.append(title.replace(".jpeg",""))
        
    return images, responses
    
def plot_image(image:pd.Series):
    fig, ax = plt.subplots(figsize=(3,3))
    ax.axis('off')
    ax.imshow(image.to_numpy().reshape(28,28),cmap="gray")
    plt.show()
            
def save_dataframes(train_X:pd.DataFrame, test_X:pd.DataFrame,
                    train_Y:pd.Series, test_Y:pd.Series):
    
    dfs_path = os.getcwd()+"/../data/dataframes" 
    if not os.path.exists(dfs_path):
        os.makedirs(dfs_path)

    train_X.to_parquet(dfs_path+"/train_X.parquet")
    train_Y.to_frame().to_parquet(dfs_path+"/train_Y.parquet")

    test_X.to_parquet(dfs_path+"/test_X.parquet")
    test_Y.to_frame().to_parquet(dfs_path+"/test_Y.parquet")
    
    
def load_dataframes(isTrain:bool)->Tuple[pd.DataFrame,pd.Series]:
    
    dfs_path = os.getcwd()+"/../data/dataframes" 
    if not os.path.exists(dfs_path):
        raise FileNotFoundError("dataframes directory does not exist. Please execute dataset notebook first.")
    
    if isTrain:
        dfs_path += "/train_"
    else:
        dfs_path += "/test_"
        
    X = pd.read_parquet(dfs_path+"X.parquet")
    Y = pd.read_parquet(dfs_path+"Y.parquet").squeeze()
    
    return X, Y

def load_test_results(base_model_name:str):
    
    if base_model_name not in ["Classifier_1", "Classifier_2", "Classifier_3", "LeNet5"]:
        raise IOError("Model not found")
    
    all_models_names = [f for f in listdir("../models/") if isfile(f"../models/{f}") and base_model_name in f]
    
    architecture_id_to_model_name = {}
    results=[]
    
    for i, name in enumerate(all_models_names):
        
        result = pd.read_parquet(f"../models/{name}")
        result.insert(0, "architecture_id", i)
        result.reset_index(names="test_id", inplace=True)
        
        results.append(result)
        
        architecture_id_to_model_name[i] = name[:-16]
    
    all_model_results = pd.concat(results, ignore_index=True)
    
    return architecture_id_to_model_name, all_model_results

def architecture_stats( all_model_results:pd.DataFrame, architecture_id_to_model_name:dict , architecture_id:int):
    
    result = all_model_results[all_model_results["architecture_id"]==architecture_id]
    
    name = architecture_id_to_model_name[architecture_id]
    
    print(f"Stats for architecture: {name} (id: {architecture_id})")
    
    print(f"mean accuracy: {result['test_accuracies'].mean():.4f} with standard error: {result['test_accuracies'].std():.4f}\n")
    
    print(f"worst accuracy: {result['test_accuracies'].min():.4f} with hyperparameters:")
    print(result.iloc[result["test_accuracies"].argmin()]
            .drop(["test_accuracies", "architecture_id","n_neurons_molt_factor","do_dropout"]),"\n")
    
    print(f"best accuracy: {result['test_accuracies'].max():.4f} with hyperparameters:")
    print(result.iloc[result["test_accuracies"].argmax()]
            .drop(["test_accuracies", "architecture_id","n_neurons_molt_factor","do_dropout"]),"\n")