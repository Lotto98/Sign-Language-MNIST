from sklearn.model_selection import GridSearchCV
from skorch import NeuralNetClassifier

from utility import load_dataframes
from data_transform import ImageDataset

from models import Classifier

import torch

import torch.optim as optim

from torch.utils.data import DataLoader

import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 64

(train_X, train_y) = load_dataframes(is_train=True)
train_dataset = ImageDataset(train_X, train_y)

X=train_X.to_numpy(dtype=np.float32).reshape(len(train_X),1,28,28)
y=train_y.to_numpy()

train_loader = DataLoader(
        dataset = train_dataset,
        batch_size = batch_size,
    )

net = NeuralNetClassifier(
    Classifier,
    max_epochs=10,
    criterion=torch.nn.CrossEntropyLoss,
    lr=0.01,
    device="cuda",
    # Shuffle training data on each epoch
    train_split=None,
    iterator_train__shuffle=False,
)
gs= GridSearchCV(net, param_grid={
                                'batch_size': [10, 20, 40, 60, 80, 100],
                                'max_epochs': [10, 50, 100],
                                'optimizer__lr': [0.001, 0.01, 0.1, 0.2, 0.3],
                                'optimizer__momentum': [0.0, 0.2, 0.4, 0.6, 0.8, 0.9],
                                }, scoring="accuracy", cv=2, verbose=3)

gs.fit(X,y)

print('SEARCH COMPLETE')
print("best score: {:.3f}, best params: {}".format(gs.best_score_, gs.best_params_))