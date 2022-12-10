import torch

# Defines a distance metric between two neural networks which is the mean squared difference between the weights of the two networks times a constant
def distance_mse(network1, network2):
    distance = 0
    total_params = 0
    for param1, param2 in zip(network1.parameters(), network2.parameters()):
        distance += torch.mean((param1 - param2) ** 2)*param1.numel()
        total_params += param1.numel()
    return distance/total_params/0.07

# Defines a null distance metric which always returns 0
def distance_null(network1, network2):
    return 0

# Defines a discrete distance metric which is 0 if the two networks are equal and 1 otherwise
def distance_discrete(network1, network2):
    for param1, param2 in zip(network1.parameters(), network2.parameters()):
        if not torch.equal(param1, param2):
            return 1
    return 0

# Returns the game matrix of various dilemmas dilemma in tensor form. Shifts the rewards to [-1, 0] to aid learning
def get_prisoners_dilemma():
    return torch.tensor([[3, 0], [5, 1]], dtype=torch.float)/5 - 1

def get_stag_hunt():
    return torch.tensor([[2, 0], [1, 1]], dtype=torch.float)/2 - 1

def get_uniform_game(action_n=2):
    return torch.rand(action_n, action_n, dtype=torch.float) - 1

def get_normal_game(action_n=2):
    return torch.randn(action_n, action_n, dtype=torch.float)/2 - 1/2

# Defines neural networks used for Agent
class Net(torch.nn.Module):
    def __init__(self, hidden_size, action_n=2, random_inputs=0):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(action_n**2 + 1 + random_inputs, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, action_n)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x