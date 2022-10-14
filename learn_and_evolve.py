import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import copy
from parameter_functions import *


# Define the agent class, which can learn and evolve
class Agent:
    # Initialize the agent with a neural network and a distance metric
    def __init__(self, network, distance_f, criterion=nn.MSELoss(), optimizer=torch.optim.SGD, lr=0.01):
        self.network = network
        self.score = 0
        self.distance_f = distance_f
        self.criterion = criterion
        self.optimizer = optimizer(self.network.parameters(), lr=lr)

# Create an environment in which RL agents interact through one-shot games and occaisionally reproduce
class Environment:
    # Initialize the environment with a list of agents and a matrix game generator
    def __init__(self, agents, game_generator):
        self.agents = agents
        self.game_generator = game_generator
    
    # Play a one-shot game between two agents
    def play_game(self, game, agent1, agent2):
        game_flattened = game.flatten()
        # Get the distance from the agents to each other (note that the distance is not necessarily symmetric)
        distance1 = agent1.distance_f(agent1.network, agent2.network)
        distance2 = agent2.distance_f(agent2.network, agent1.network)
        # Combine the game and the distances into a single tensor for each agent
        input1 = torch.cat((game_flattened, torch.tensor([distance1])))
        input2 = torch.cat((game_flattened, torch.tensor([distance2])))
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
        loss1 = agent1.criterion(output1[action1], torch.tensor([reward1]))
        loss2 = agent2.criterion(output2[action2], torch.tensor([reward2]))
        # Backpropagate the loss of each agent's network
        loss1.backward()
        loss2.backward()
        # Update the parameters of each agent's network
        agent1.optimizer.step()
        agent2.optimizer.step()

    # Has all of the agents play one-shot games a certain number of times, then updates the set of agents
    def round(self, num_games, death_rate, print_average_score=False):
        for _ in range(num_games):
            random.shuffle(self.agents)
            # Iterate through every other agent
            for i in range(0, len(self.agents), 2):
                # Play a game between the current agent and the next agent
                self.play_game(self.game_generator(), self.agents[i], self.agents[i+1])
        # Sort the agents by score
        self.agents.sort(key=lambda agent: agent.score, reverse=True)
        # Print the average score(?) and zero out the scores
        if print_average_score:
            print(sum(agent.score for agent in self.agents) / len(self.agents))
        map(lambda agent: setattr(agent, 'score', 0), self.agents)
        death_count = int(len(self.agents) * death_rate)
        for i in range(death_count):
            self.agents[-i-1] = Agent(copy.deepcopy(self.agents[i].network), self.agents[i].distance_f)

    # Run the environment for a certain number of rounds
    def run(self, num_rounds, num_games, death_rate, print_average_score=True):
        for i in range(num_rounds):
            print(f'Round {i+1}:', end=' ')
            self.round(num_games, death_rate, print_average_score)

def main():
    # Create a list of agents
    agents = [Agent(Net(hidden_size=16), distance_mse) for _ in range(100)]
    # Create an environment
    env = Environment(agents, get_prisoners_dilemma, criterion=nn.MSELoss(), optimizer=torch.optim.SGD)
    # Run the environment
    env.run(100, 100, 0.1)

if __name__ == '__main__':
    main()