import torch

# Defines a distance metric between two neural networks which is the mean squared difference between the weights of the two networks
def distance_mse(network1, network2):
    distance = 0
    total_params = 0
    for param1, param2 in zip(network1.parameters(), network2.parameters()):
        distance += torch.mean((param1 - param2) ** 2)*param1.numel()
        total_params += param1.numel()
    return distance/total_params

# Defines a null distance metric which always returns 0
def distance_null(network1, network2):
    return 0

# Returns the game matrix of various dilemmas dilemma in tensor form. Shifts the rewards to [-1, 0] to aid learning
def get_prisoners_dilemma():
    return torch.tensor([[[3, 0], [5, 1]], [[3, 5], [0, 1]]], dtype=torch.float)/6 - 1

def get_stag_hunt():
    return torch.tensor([[[2, 0], [1, 1]], [[2, 1], [0, 1]]], dtype=torch.float)/3 - 1

def get_uniform_game():
    return torch.rand(2, 2, 2, dtype=torch.float) - 1

def get_normal_game():
    return torch.randn(2, 2, 2, dtype=torch.float) - 1

# Defines neural networks used for Agent
class Net(torch.nn.Module):
    def __init__(self, hidden_size, action_n=2):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(2*action_n**2 + 1, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, action_n)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x