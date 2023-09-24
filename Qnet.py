import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import random
class Qnet(nn.Module):  # Qnet
    def __init__(self, input_layer, output_layer):
        super(Qnet, self).__init__()
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.fc1 = nn.Linear(self.input_layer, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, self.output_layer)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, self.output_layer-1)
        else:
            return out.argmax().item()

    def select_action(self, obs, epsilon):
        out = self.forward(obs)
        return out.argmax().item(), out
