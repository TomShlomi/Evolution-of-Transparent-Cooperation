import torch

# Defines a distance metric between two neural networks which is the mean squared difference between the weights of the two networks
def distance_f(network1, network2):
    distance = 0
    total_params = 0
    for param1, param2 in zip(network1.parameters(), network2.parameters()):
        distance += torch.mean((param1 - param2) ** 2)*param1.numel()
        total_params += param1.numel()
    return distance/total_params

# Returns the game matrix of the prisoner's dilemma in tensor form. Shifts the rewards to [-1, 0] to aid learning
def get_prisoners_dilemma():
    return torch.tensor([[[3, 0], [5, 1]], [[3, 5], [0, 1]]], dtype=torch.float)/6 - 1