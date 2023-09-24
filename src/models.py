import torch.nn as nn
import torch.optim as optim

from utility import load_dataframes

from sklearn.model_selection import KFold

from data_transform import ImageDataset

from torch.utils.data import DataLoader, random_split

import torch

loss = nn.CrossEntropyLoss()

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.Conv1 = nn.Sequential(
        nn.Conv2d(1, 4, 5), # 24x24x4 
        nn.MaxPool2d(2), # 12x12x4
        nn.ReLU()
        )
        self.Conv2 = nn.Sequential(
        nn.Conv2d(4, 8, 5), # 8x8x8
        nn.MaxPool2d(2),  # 4x4x8
        nn.ReLU()
        )
        
        self.Linear1 = nn.Linear(8 * 4 * 4, 64)
        self.Linear2 = nn.Linear(64, 32)
        self.Linear3 = nn.Linear(32, 25)
        
    def forward(self, x):
        x = self.Conv1(x)
        x = self.Conv2(x)
        x = x.view(x.size(0), -1) #flatten
        x = self.Linear1(x)
        x = self.Linear2(x)
        x = self.Linear3(x)
        return x


# Define the training function
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    
    train_loss=0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        
        output = model(data.to(device))
        
        l = loss(output.to(device), target.to(device))
        
        train_loss += l.item() 
        #print(output)
        
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        
    train_loss /= len(train_loader.dataset)
    
    return train_loss

# Define the training function
def evaluate(model, device, val_loader, epoch):
    model.eval()
    
    val_loss = 0
    with torch.no_grad():
        for data, target in val_loader:
            
            output = model(data.to(device))
        
            l = loss(output.to(device), target.to(device))
            
            val_loss += l.item()

        val_loss /= len(val_loader.dataset)
    
        return val_loss

def test(model, device, test_loader):
    model.eval()
    
    accuracy = 0
    with torch.inference_mode():
        for data, target in test_loader:
            
            output = model(data.to(device))
        
            output_prob = torch.softmax(output, dim=1)
            prediction = torch.argmax(output_prob, dim=1)
            
            accuracy += torch.sum(prediction==target).item()
            
        accuracy = accuracy/len(test_loader.dataset)
    
    return accuracy

def spit_train(train_data, perc_val_size):
    
    train_size = len(train_data)
    val_size = int((train_size * perc_val_size) // 100)
    train_size -= val_size

    return random_split(train_data, [int(train_size), int(val_size)]) #train_data, val_data 

def full_training():
    # Define the device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 64

    (train_X, train_y) = load_dataframes(is_train=True)
    train_dataset = ImageDataset(train_X, train_y, False)

    train_dataset, val_dataset = spit_train(train_dataset,20)

    # Define the data loaders
    train_loader = DataLoader(
        dataset = train_dataset,
        batch_size = batch_size,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
    )

    # Initialize the model and optimizer
    model = Classifier().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    # Train the model on the current fold
    for epoch in range(1, 150):
        train_loss = train(model, device, train_loader, optimizer, epoch)
        val_loss = evaluate(model, device, val_loader, epoch)
        print(epoch, train_loss, val_loss)


    (test_X, test_y) = load_dataframes(is_train=False)

    test_dataset = ImageDataset(test_X, test_y, False)

    test_loader = DataLoader(
        dataset = test_dataset,
        batch_size = batch_size,
    )

    accuracy = test(model,device,test_loader)

    print(accuracy)

full_training()

"""
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
    optimizer = optim.SGD(model.parameters(), lr=0.2, momentum=0.5)

    # Train the model on the current fold
    for epoch in range(1, 50):
        train_loss = train(model, device, train_loader, optimizer, epoch)
        print(epoch, train_loss)
    
    # Evaluate the model on the test set
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data.to(device))
        
            l = loss(output.to(device), target.to(device))
            
            val_loss += l.item()
            import numpy as np
            print(np.argmax(torch.softmax(output,dim=1)[1].cpu().numpy()),target[1])

    val_loss /= len(val_loader.dataset)

    # Print the results for the current fold
    print(f"Test set: Average loss: {val_loss:.4f}")
"""