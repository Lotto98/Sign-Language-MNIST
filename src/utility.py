import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import gc

response_conv={n:chr(n+65) for n in range(0,26)}

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
            ax.set_title(response_conv[k])
            
            if k!=9 and k!=25: #impossible to distinguish J and Z
                
                indexes = train_Y[train_Y==k].index
                i = indexes[ran % len(indexes)]
                
                to_show=np.reshape(train_X.iloc[i].to_numpy(),(28,28))
            else:
                to_show=np.zeros((28,28))
            
            ax.imshow(to_show,cmap='gray')