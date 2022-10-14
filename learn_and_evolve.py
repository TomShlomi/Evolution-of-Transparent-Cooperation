import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Define the agent class, which can learn and evolve
class Agent:
    # Initialize the agent with a neural network and a distance metric
    def __init__(self, network, distance_f):
        self.network = network
        self.score = 0
        self.distance_f = distance_f


# Create an environment in which RL agents interact through one-shot games and occaisionally reproduce
class Environment:
    # Initialize the environment with a list of agents and a matrix game generator
    def __init__(self, agents, game_generator, criterion=nn.MSELoss(), optimizer=torch.optim.SGD):
        self.agents = agents
        self.game_generator = game_generator
        self.criterion = criterion
        self.optimizer = optimizer
    
    # Play a one-shot game between two agents
    def play_game(self, game, agent1, agent2)
        game_flattened = game.flatten()
        # Get the distance from the agents to each other (note that the distance is not necessarily symmetric)
        distance1 = agent1.distance_f(agent1.network, agent2.network)
        distance2 = agent2.distance_f(agent2.network, agent1.network)
        # Combine the game and the distances into a single tensor for each agent
        input1 = torch.cat((game_flattened, distance1))
        input2 = torch.cat((game_flattened, distance2))
        # Get the output of each agent's network
        output1 = agent1.network(input1)
        output2 = agent2.network(input2)
        # Get the action of each agent
        action1 = torch.argmax(output1)
        action2 = torch.argmax(output2)
        # Get the reward of each agent
        reward1 = game[0, action1, action2]
        reward2 = game[1, action2, action1]
        # Update the score of each agent
        agent1.score += reward1
        agent2.score += reward2
        # Zero the gradients of each agent's network
        agent1.network.zero_grad()
        agent2.network.zero_grad()
        # Calculate the loss of each agent's network
        loss1 = self.criterion(output1[action1], torch.tensor([reward1]))
        loss2 = self.criterion(output2[action2], torch.tensor([reward2]))
        # Backpropagate the loss of each agent's network
        loss1.backward()
        loss2.backward()
        # Update the parameters of each agent's network
        self.optimizer.step()