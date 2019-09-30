import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pearl.utils.misc import hidden_init

class Actor(nn.Module):
    def __init__(self,
                    state_dim,
                    action_dim,
                    max_action = 1.,
                    batch_norm = False,
                    return_probs = False,
                    fc1_out = 300,
                    fc2_out= 400):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, fc1_out)
        self.fc2 = nn.Linear(fc1_out, fc2_out)
        self.fc3 = nn.Linear(fc2_out, action_dim)
        self.bn1 = nn.BatchNorm1d(fc1_out)
        self.max_action = max_action
        self.return_probs = return_probs
        self.batch_norm = batch_norm
        self.reset_parameters()
    
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-1e-3, 1e-3)

    def forward(self, x):
        if self.batch_norm:
            x = F.relu(self.bn1(self.fc1(x)))
        else:
            x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        if self.return_probs:
            return F.sigmoid(logits)
        else:
            return self.max_action*torch.tanh(logits)


class Critic(nn.Module):
    def __init__(self,
                    state_dim,
                    action_dim,
                    batch_norm = False,
                    fc1_out = 300,
                    fc2_out= 400):
        super(Critic, self).__init__()
        self.fc_xs = nn.Linear(state_dim + action_dim, fc1_out)
        self.fc1 = nn.Linear(fc1_out, fc2_out)
        self.fc2 = nn.Linear(fc2_out, 1)
        self.bn1 = nn.BatchNorm1d(fc1_out)
        self.batch_norm = batch_norm
        self.reset_parameters()
    
    def reset_parameters(self):
        self.fc_xs.weight.data.uniform_(*hidden_init(self.fc_xs))
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(-1e-3, 1e-3)

    def forward(self, state, action):
        if self.batch_norm:
            xs = F.relu(self.bn1(self.fc_xs(torch.cat((state, action), dim=1))))
        else:
            xs = F.relu(self.fc_xs(torch.cat((state, action), dim=1)))
        x = F.relu(self.fc1(xs))
        return self.fc2(x)