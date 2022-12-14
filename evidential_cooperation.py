from os import environ
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import copy
import matplotlib.pyplot as plt
from parameter_functions import distance_mse, distance_null, distance_discrete, get_prisoners_dilemma, get_stag_hunt, get_uniform_game, get_normal_game, Net


# Define the agent class, which can learn and evolve
class Agent:
    # Initialize the agent with a neural network and a distance metric
    def __init__(self, network, distance_f, criterion=nn.MSELoss(), optimizer=torch.optim.SGD, lr=0.001):
        self.network = network
        self.score = 0
        self.distance_f = distance_f
        self.criterion = criterion
        self.optimizer = optimizer(self.network.parameters(), lr=lr)

# Create an environment in which RL agents interact through one-shot games and occaisionally reproduce
class Environment:
    # Initialize the environment with a list of agents and a matrix game generator
    def __init__(self, agents, game_generator, random_inputs=0):
        self.agents = agents
        self.game_generator = game_generator
        self.random_inputs = random_inputs
        self.same_distance_fs = True
        for agent in self.agents:
            if agent.distance_f != self.agents[0].distance_f:
                self.same_distance_fs = False
                break
    
    # Play a one-shot game between two agents
    def play_game(self, game, agent1, agent2):
        game_flattened = game.flatten()
        # Get the distance from the agents to each other (note that the distance is not necessarily symmetric)
        distance1 = agent1.distance_f(agent1.network, agent2.network)
        distance2 = distance1 if self.same_distance_fs else agent2.distance_f(agent2.network, agent1.network)
        # Generate shared normal noise
        noise = torch.randn(self.random_inputs)
        # Combine the game and the distances into a single tensor for each agent
        input1 = torch.cat((game_flattened, torch.tensor([distance1]), noise))
        input2 = torch.cat((game_flattened, torch.tensor([distance2]), noise))
        # Get the output of each agent's network
        output1 = agent1.network(input1)
        output2 = agent2.network(input2)
        # Get the action of each agent
        action1 = torch.argmax(output1)
        action2 = torch.argmax(output2)
        # Get the reward of each agent
        reward1 = game[action1, action2]
        reward2 = game[action2, action1]
        # Update the score of each agent
        agent1.score += reward1
        agent2.score += reward2
        # Zero the gradients of each agent's network
        agent1.network.zero_grad()
        agent2.network.zero_grad()
        # Calculate the loss of each agent's network
        loss1 = agent1.criterion(output1[action1], reward1)
        loss2 = agent2.criterion(output2[action2], reward2)
        # Backpropagate the loss of each agent's network
        loss1.backward()
        loss2.backward()
        # Update the parameters of each agent's network
        agent1.optimizer.step()
        agent2.optimizer.step()

    # Has all of the agents play one-shot games a certain number of times, then updates the set of agents
    def round(self, num_games, death_rate):
        for _ in range(num_games):
            random.shuffle(self.agents)
            # Iterate through every other agent
            for i in range(0, len(self.agents), 2):
                # Play a game between the current agent and the next agent
                self.play_game(self.game_generator(), self.agents[i], self.agents[i+1])
        # Sort the agents by score
        self.agents.sort(key=lambda agent: agent.score, reverse=True)
        # Calculate the average score of the agents
        average_score = sum([agent.score/num_games for agent in self.agents]) / len(self.agents)
        # Zero the scores of all of the agents
        for agent in self.agents:
            agent.score = 0
        # Kill the worst agents and replace them with copies of the best agents
        death_count = int(len(self.agents) * death_rate)
        for i in range(death_count):
            self.agents[-i-1] = Agent(copy.deepcopy(self.agents[i].network), self.agents[i].distance_f)
        # Return the average score of the agents
        return average_score.item()

    # Run the environment for a certain number of rounds
    def run(self, num_rounds, num_games, death_rate):
        average_scores = []
        for _ in range(num_rounds):
            average_scores.append(self.round(num_games, death_rate))
        return average_scores

# Get average scores
def get_average_scores(agents_generator, game_generator, death_rate=0.0, num_trials=1000, num_rounds=1000, num_games=1, random_inputs=0, verbose=False):
    average_scores = [0] * num_rounds
    nets = None

    for i in range(num_trials):
        agents, nets = agents_generator()

        # Create and run the environment
        env = Environment(agents=agents, game_generator=game_generator, random_inputs=random_inputs)
        scores = env.run(num_rounds=num_rounds, num_games=num_games, death_rate=death_rate)
        # Add the scores divided by the number of trials to the average scores
        for j in range(len(scores)):
            average_scores[j] += scores[j]/num_trials
        if verbose:
            print("Trial " + str(i+1) + " average: " + str(sum(scores)/len(scores)))
    return average_scores, nets

# Test a list of nets to see how their distance input affects their odds of cooperating
def test_net_cooperativity(nets, game_generator, random_inputs, randomize=False, range=(0, 2)):
    # Get a linspace over range
    linspace = torch.tensor(np.linspace(range[0], range[1], 100), dtype=torch.float)
    cooperativities = []
    for net in nets:
        outputs = []
        for i in linspace:
            noise = torch.randn(random_inputs) if randomize else torch.zeros(random_inputs)
            input = torch.cat((game_generator().flatten(), torch.tensor([i]), noise))
            outputs.append(net(input))
        cooperativities.append(outputs)
    return cooperativities
