import torch.nn as nn

class LeNet5(nn.Module):
    def __init__(self, num_classes=25):
        super(LeNet5, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        
        self.fc = nn.Linear(400, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out
    
class Classifier_3(nn.Module):
    def __init__(self, n_neurons = 512):
        super(Classifier_3, self).__init__()
        
        self.Conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 5), #24x24x32
            nn.MaxPool2d(2), #12x12x32
            nn.ReLU()
        )
        self.Conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5), #8x8x64
            nn.MaxPool2d(2), #4x4x64
            nn.ReLU()
        )
        self.Conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3), #2x2x128
            nn.MaxPool2d(2), #1x1x128
            nn.ReLU()
        )
        
        self.dropout=nn.Dropout()
        
        self.Linear1 = nn.Linear(128, n_neurons)
        self.relu = nn.ReLU()
        self.Linear2 = nn.Linear(n_neurons, 32)
        self.relu1 = nn.ReLU()
        self.Linear3 = nn.Linear(32, 25)
        
    def forward(self, x):
        x = self.Conv1(x)
        x = self.Conv2(x)
        x = self.Conv3(x)
        x = x.view(x.size(0), -1) #flatten
        x = self.dropout(x)
        x = self.Linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.Linear2(x)
        x = self.relu(x)
        x = self.Linear3(x)
        return x

class Classifier_2(nn.Module):
    def __init__(self, n_neurons = 512):
        super(Classifier_2, self).__init__()
        self.Conv1 = nn.Sequential(
        nn.Conv2d(1, 16, 5), # 24x24x16 
        nn.MaxPool2d(2), # 12x12x16
        nn.ReLU()
        )
        
        self.Conv2 = nn.Sequential(
        nn.Conv2d(16, 32, 5), # 8x8x32
        nn.MaxPool2d(2),  # 4x4x32
        nn.ReLU()
        )
        
        self.Linear1 = nn.Linear(32 * 4 * 4, n_neurons)
        self.Linear2 = nn.Linear(n_neurons, 32)
        self.Linear3 = nn.Linear(32, 25)
        
    def forward(self, x):
        x = self.Conv1(x)
        x = self.Conv2(x)
        x = x.view(x.size(0), -1) #flatten
        x = self.Linear1(x)
        x = self.Linear2(x)
        x = self.Linear3(x)
        return x