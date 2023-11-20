import torch.nn as nn
import torch

from typing import List, Tuple

from torchsummary import summary

class Model(nn.Module):
    def __init__(self, model_input_dim:Tuple[int,int], device: torch.device) -> None:
        super(Model, self).__init__()
        
        self.model_input_dim = (1, model_input_dim[0], model_input_dim[1])
        self.device = device
        
    def print_architecture(self, print_architecture_spec:bool)->str:
        
        architecture_name = type(self).__name__

        if print_architecture_spec:
            architecture_name += f" (n_neurons_molt_factor={self.n_neurons_molt_factor}, do_dropout={self.do_dropout_list})"
        
        print(f"name: {architecture_name}")
        summary(self, self.model_input_dim, device=self.device)
        
        return architecture_name
        
class LeNet5(Model):
    def __init__(self, device: torch.device):
        super(LeNet5, self).__init__( (32,32), device)
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
        self.fc2 = nn.Linear(84, 24)
        
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

class Classifier_1(Model):
    def __init__(self, device: torch.device, input:int=28*28, n_neurons_molt_factor:float = 1, do_dropout:List[str]=[]):
        super(Classifier_1, self).__init__((28,28), device=device)
        
        self.n_neurons_molt_factor=n_neurons_molt_factor
        self.do_dropout_list=do_dropout
        
        n_neurons = int(input * n_neurons_molt_factor)
        
        self.FC1 = nn.Sequential(
            nn.Linear(input, n_neurons),
            nn.ReLU(),
            Classifier_1.do_dropout("FC1" in do_dropout)
        )
        
        self.FC2 = nn.Sequential(
            nn.Linear(n_neurons, 32),
            nn.ReLU(),
        )
        
        self.Linear3 = nn.Linear(32, 24)
        
    def forward(self, x):
        x = x.view(x.size(0), -1) #flatten
        x = self.FC1(x)
        x = self.FC2(x)
        x = self.Linear3(x)
        return x
    
    @staticmethod
    def do_dropout(condition:bool):
        if condition: 
            return nn.Dropout()
        else:
            return nn.Identity()
        
class Classifier_2(Classifier_1):
    def __init__(self, device: torch.device, n_neurons_molt_factor:float = 1, do_dropout:List[str]=[], ):
        super().__init__(device, 4 * 4 * 32, n_neurons_molt_factor, do_dropout)
        
        self.Conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 5), # 24x24x16 
            nn.MaxPool2d(2), # 12x12x16
            nn.ReLU()
        )
        
        self.Conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5), # 8x8x32
            nn.MaxPool2d(2),  # 4x4x32
            nn.ReLU(),
            super().do_dropout("Conv2" in do_dropout)
        )
        
    def forward(self, x):
        
        x = self.Conv1(x)
        x = self.Conv2(x)
        
        return super().forward(x)

class Classifier_3(Classifier_1):
    def __init__(self, device: torch.device, n_neurons_molt_factor:float = 1, do_dropout:List[str]=[]):
        super().__init__(device, 128, n_neurons_molt_factor, do_dropout)
        
        self.Conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 5), #24x24x32
            nn.MaxPool2d(2), #12x12x32
            nn.ReLU()
        )
        
        self.Conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5), #8x8x64
            nn.MaxPool2d(2), #4x4x64
            nn.ReLU(),
            super().do_dropout("Conv2" in do_dropout)
        )
        
        self.Conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3), #2x2x128
            nn.MaxPool2d(2), #1x1x128
            nn.ReLU(),
            super().do_dropout("Conv3" in do_dropout)
        )
        
    def forward(self, x):
        
        x = self.Conv1(x)
        x = self.Conv2(x)
        x = self.Conv3(x)
        
        return super().forward(x)