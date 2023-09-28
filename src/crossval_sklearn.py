
from sklearn.experimental import enable_halving_search_cv

from sklearn.model_selection import HalvingRandomSearchCV
from skorch import NeuralNetClassifier

from utility import load_dataframes
from data_transform import ImageDataset

from models import Classifier_3

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
    Classifier_3,
    criterion = torch.nn.CrossEntropyLoss,
    lr=0.01,
    device="cuda",
    # Shuffle training data on each epoch
    train_split=False,
    iterator_train__shuffle=False,
)
gs = HalvingRandomSearchCV(net, param_distributions={
                                'batch_size': [10, 20, 40, 60, 80, 100],
                                'max_epochs': [10, 50, 100],
                                'optimizer__lr': [0.001, 0.01, 0.1, 0.2, 0.3],
                                'optimizer__momentum': [0.0, 0.2, 0.4, 0.6, 0.8, 0.9],
                                }, scoring="accuracy", cv=5, verbose=3, refit=True)

"""
gs.fit(X,y)

print('SEARCH COMPLETE')
print("best score: {:.3f}, best params: {}".format(gs.best_score_, gs.best_params_))

net = gs.best_estimator_
"""

net.fit(X,y)

(test_X, test_y) = load_dataframes(is_train=False)

test_dataset = ImageDataset(test_X, test_y)

X_test=test_X.to_numpy(dtype=np.float32).reshape(len(test_X),1,28,28)
y_test=test_y.to_numpy()

y_pred = net.predict(X_test)

from sklearn.metrics import accuracy_score

print(y_pred)
print(y_test)

print(accuracy_score(y_test, y_pred))

