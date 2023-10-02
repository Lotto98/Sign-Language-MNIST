from NeuralNetwork import NeuralNetwork

from models import Classifier_3, LeNet5

import torch.optim as optim

model = Classifier_3()
optimizer = optim.Adam(model.parameters(),lr=0.0001, amsgrad=True)

net = NeuralNetwork(model,optimizer,"Classifier_3", model_input_dim = (28,28) )

#net.load_model()

net.full_training()