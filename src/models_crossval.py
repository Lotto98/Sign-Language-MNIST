from utility import load_dataframes

from models import Classifier, train, evaluate, test

from data_transform import ImageDataset

from torch.utils.data import DataLoader, SubsetRandomSampler

from sklearn.model_selection import KFold

import torch.optim as optim
import torch.nn as nn

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loss = nn.CrossEntropyLoss()

batch_size = 64
k_folds=5

(train_X, train_y) = load_dataframes(is_train=True)
train_dataset = ImageDataset(train_X, train_y, False)

# Initialize the k-fold cross validation
kf = KFold(n_splits=k_folds, shuffle=True)

for fold, (train_idx, val_idx) in enumerate(kf.split(train_X)):
    print(f"Fold {fold + 1}")
    print("-------")

    # Define the data loaders for the current fold
    train_loader = DataLoader(
        dataset = train_dataset,
        batch_size = batch_size,
        sampler = SubsetRandomSampler(train_idx.tolist()),
    )
    
    val_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        sampler = SubsetRandomSampler(val_idx.tolist()),
    )
    
    # Initialize the model and optimizer
    model = Classifier().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    # Train the model on the current fold
    for epoch in range(1, 50):
        train_loss = train(model, device, train_loader, optimizer, epoch)
        eval_loss = evaluate(model, device, val_loader, epoch)
        print(epoch, train_loss, eval_loss)
    
    (test_X, test_y) = load_dataframes(is_train=False)

    test_dataset = ImageDataset(test_X, test_y, False)

    test_loader = DataLoader(
        dataset = test_dataset,
        batch_size = batch_size,
    )

    accuracy = test(model,device,test_loader)

    print(accuracy)
