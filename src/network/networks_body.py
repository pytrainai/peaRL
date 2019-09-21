import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.misc import hidden_init

class ConvBody(nn.Module):
    def __init__(self, in_channels=3):
        super(ConvBody, self).__init__()
        self.feature_dim = 512
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(7 * 7 * 64, self.feature_dim)
    
    def reset_parameters(self):
        self.conv1.weight.data.uniform_(*hidden_init(self.conv1))
        self.conv2.weight.data.uniform_(*hidden_init(self.conv2))
        self.conv3.weight.data.uniform_(*hidden_init(self.conv3))
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = y.view(y.size(0), -1)
        y = F.relu(self.fc1(y))
        return y

class CriticConvBody(nn.Module):
    def __init__(self, in_channels=3):
        super(CriticConvBody, self).__init__()
        self.feature_dim = 512
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(7 * 7 * 64, self.feature_dim)
    
    def reset_parameters(self):
        self.conv1.weight.data.uniform_(*hidden_init(self.conv1))
        self.conv2.weight.data.uniform_(*hidden_init(self.conv2))
        self.conv3.weight.data.uniform_(*hidden_init(self.conv3))
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = y.view(y.size(0), -1)
        y = F.relu(self.fc1(y))
        return y

class LinearBody(nn.Module):
    def __init__(self, input_dim, out_dim = 300, fc1_units = 400):
        super(LinearBody, self).__init__()
        self.fc1 = nn.Linear(input_dim, fc1_units)
        self.fc2 = nn.Linear(fc1_units, out_dim)
        #self.bn1 = nn.BatchNorm1d(fc1_units)
        self.feature_dim = out_dim
        self.reset_parameters()
    
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))

    def forward(self, x):
        #y = F.relu(self.bn1(self.fc1(x)))
        y = F.relu(self.fc1(x))
        return F.relu(self.fc2(y))
    
class CriticLinearBody(nn.Module):
    def __init__(self, state_dim, action_dim, out_dim = 300, fc1_units = 400):
        super(CriticLinearBody, self).__init__()
        self.fc_xs = nn.Linear(state_dim + action_dim, fc1_units)
        self.fc1 = nn.Linear(fc1_units, out_dim)
        #self.bn1 = nn.BatchNorm1d(fc1_units)
        self.feature_dim = out_dim
        self.reset_parameters()
    
    def reset_parameters(self):
        self.fc_xs.weight.data.uniform_(*hidden_init(self.fc_xs))
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))

    def forward(self, x, action):
        #xs = torch.cat([F.relu(self.bn1(self.fc_x(x))), self.fc_a(action)], dim=1)
        xs = F.relu(self.fc_xs(torch.cat((x, action), dim=1)))
        return F.relu(self.fc1(xs))