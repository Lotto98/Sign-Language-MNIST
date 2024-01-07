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


response_transform={n:chr(n+65) for n in range(0,9)} | {n:chr(n+66) for n in range(9,24)}

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

    for k, ax in enumerate(axs):
        ax.axis("off")
        
        if k<25:
            
            if k==9:
                ax.set_title("J")
                to_show=np.zeros((28,28))
            else:
                if k>9:
                    k=k-1
                
                ax.set_title(response_transform[k])

                indexes = train_Y[train_Y==k].index
                i = indexes[ran % len(indexes)]
                
                to_show=np.reshape(train_X.iloc[i].to_numpy(),(28,28)) # type: ignore
                
            ax.imshow(to_show,cmap='gray')
            
        if k==25:
            ax.set_title("Z")
            to_show=np.zeros((28,28))
            ax.imshow(to_show,cmap='gray') # type: ignore

def sample_image(index:int=-1):
    
    test_X, test_y  = load_dataframes(isTrain=False)
    
    if index==-1:
        index = randint(0, len(test_X)-1)
        print(f"Sampled image with index {index}")
    
    image = test_X.iloc[index]
    response = test_y.iloc[index]
    
    plot_image(image)
    print(f"True label: {response_transform[response]}")
    
    return image, response

"""
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
    titles.sort()
    
    for title in titles:
        if isfile(os.getcwd()+'/../other_data/'+title):
            images.append(load_my_image(title))
            responses.append(title.replace(".jpg",""))
        
    return images, responses
"""

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
    
    #all_models_names = sorted(all_models_names)
    
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

def architectures_analysis( all_model_results:pd.DataFrame, architecture_id_to_model_name:dict ):
    
    means=[]
    
    for architecture_id in sorted(architecture_id_to_model_name.keys()):
        
        result = all_model_results[all_model_results["architecture_id"]==architecture_id]
        
        means.append( result['test_accuracies'].mean() )
        
    best_architecture = np.argwhere(means == np.nanmax(means)).flatten().tolist()
    worst_architecture = np.argwhere(means == np.nanmin(means)).flatten().tolist()
    
    print(f"best architecture is id: {best_architecture}")
    print(f"worst architecture is id: {worst_architecture}")

def architecture_stats( all_model_results:pd.DataFrame, architecture_id_to_model_name:dict , architecture_id:int, latex:bool=False):
    
    result = all_model_results[all_model_results["architecture_id"]==architecture_id]
    
    if not latex:
        name = architecture_id_to_model_name[architecture_id]
        
        print(f"stats for architecture: {name} (id: {architecture_id})\n")
        
        print(f"mean accuracy: {result['test_accuracies'].mean():.4f} with SD: {result['test_accuracies'].std():.4f}\n")
        
        print(f"best accuracy: {result['test_accuracies'].max():.4f}")
        print(result.iloc[result["test_accuracies"].argmax()]
                .drop(["test_accuracies", "architecture_id","n_neurons_molt_factor","do_dropout"], errors="ignore"),"\n")
        
        print(f"worst accuracy: {result['test_accuracies'].min():.4f}")
        print(result.iloc[result["test_accuracies"].argmin()]
                .drop(["test_accuracies", "architecture_id","n_neurons_molt_factor","do_dropout"], errors="ignore"),"\n")
        
        print("\n\n")
    else:
        
        itemize="{itemize}"
        
        latex_out = f"""
        \\item Architectural features:
            \\begin{itemize}
                \\item hidd. neurons molt. factor: {result["n_neurons_molt_factor"].iloc[0]}, 
                \\item dropout after: {result["do_dropout"].iloc[0]}
            \\end{itemize}
            Test accuracy results:
            \\begin{itemize}
                \\item Mean accuracy of {result['test_accuracies'].mean():.4f} with SD of {result['test_accuracies'].std():.4f}
                \\item Best accuracy: {result['test_accuracies'].max():.4f}
                \\item Worst accuracy: {result['test_accuracies'].min():.4f}
            \\end{itemize}
        """
        print(latex_out)

def remove_outliers(df:pd.DataFrame, quant1:float=0.25, quant2:float=0.75):
    Q1 = df['test_accuracies'].quantile(quant1)
    Q3 = df['test_accuracies'].quantile(quant2)
    IQR = Q3 - Q1    #IQR is interquartile range. 

    filter = (df['test_accuracies'] >= Q1 - 1.5 * IQR) & (df['test_accuracies'] <= Q3 + 1.5 *IQR)
    return df.loc[filter], df.loc[~filter]

def plot_hyper(all_results:pd.DataFrame, plots_dimensions:Tuple[int,int]=(20,10), isLeNet5:bool=False, remove_outliers:bool=False, width_ratios:list=[1, 1]):
    
    #all_results, outliers=remove_outliers(all_results,quant1=out_quantiles[0],quant2=out_quantiles[1])
    
    if not isLeNet5:
        
        fig,axs=plt.subplots(1,2,figsize=plots_dimensions, gridspec_kw={'width_ratios': width_ratios})
        
        all_results_sorted = all_results.sort_values(by="do_dropout")
        all_results_sorted.boxplot(column ="test_accuracies", by="do_dropout", ax=axs[0], fontsize=11, showfliers = not remove_outliers)
        
        all_results.boxplot(column ="test_accuracies", by="n_neurons_molt_factor", ax=axs[1], fontsize=11, showfliers = not remove_outliers)
        
        axs[0].set_ylabel("test accuracy", fontsize=13)
    
        axs[0].set_xlabel("dropout after", fontsize=13)
        axs[1].set_xlabel("hidden neurons molt. factor", fontsize=13)
        fig.tight_layout()
        
        fig.suptitle("")
        axs[0].set_title("Boxplot grouped by 'dropout after'", fontsize=16)
        axs[1].set_title("Boxplot grouped by 'hidden neurons molt. factor'", fontsize=16)
    
    
    fig,axs=plt.subplots(1,2,figsize=plots_dimensions)
    all_results.boxplot(column ="test_accuracies", by="lr", ax=axs[0], fontsize=11, showfliers = not remove_outliers)
    all_results.boxplot(column ="test_accuracies", by="batch_size", ax=axs[1], fontsize=11, showfliers = not remove_outliers)
    
    axs[0].set_ylabel("test accuracy", fontsize=13)
    
    axs[0].set_xlabel("learning rate", fontsize=13)
    axs[1].set_xlabel("batch size", fontsize=13)
    fig.tight_layout()
    
    fig.suptitle("")
    axs[0].set_title("Boxplot grouped by 'learning rate'", fontsize=16)
    axs[1].set_title("Boxplot grouped by 'batch size'", fontsize=16)
    
    
    fig,axs=plt.subplots(1,2,figsize=plots_dimensions)
    all_results.boxplot(column =["test_accuracies"], by="patience", ax=axs[0], fontsize=11, showfliers = not remove_outliers)
    all_results.boxplot(column =["test_accuracies"], by="data_augmentation_perc", ax=axs[1], fontsize=11, showfliers = not remove_outliers)
    
    axs[0].set_ylabel("test accuracy", fontsize=13)
    
    axs[0].set_xlabel("patience", fontsize=13)
    axs[1].set_xlabel("data augmentation percentage", fontsize=13)
    fig.tight_layout()
    
    fig.suptitle("")
    axs[0].set_title("Boxplot grouped by 'patience'", fontsize=13)
    axs[1].set_title("Boxplot grouped by 'data augmentation percentage'", fontsize=16)
    
    fig,ax=plt.subplots(figsize=(plots_dimensions[0]/2, plots_dimensions[1]))
    all_results.boxplot(column =["test_accuracies"], by="optimizer", ax=ax, fontsize=11, showfliers = not remove_outliers)
    
    ax.set_ylabel("test accuracy", fontsize=13)
    
    ax.set_xlabel("optimizer", fontsize=13)
    fig.tight_layout()
    
    fig.suptitle("")
    ax.set_title("Boxplot grouped by 'optimizer'", fontsize=16)