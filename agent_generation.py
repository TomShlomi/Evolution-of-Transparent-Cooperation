import copy
from parameter_functions import Net
from evidential_cooperation import Agent

# Returns a list of agents, some of which share the same network
def generate_twins(distance_f, net_n, agents_per_net, lr=0.001, random_inputs=0):
    nets = [Net(hidden_size=16, random_inputs=random_inputs) for _ in range(net_n)]
    return [Agent(net, distance_f=distance_f, lr=lr) for net in nets for _ in range(agents_per_net)], nets

# Returns a list of agents, some of which initially share the same network, but diverge over time
def diverging_twins(distance_f, net_n, agents_per_net, lr=0.001, random_inputs=0):
    nets = [Net(hidden_size=16, random_inputs=random_inputs) for _ in range(net_n)]
    for _ in range(agents_per_net - 1):
        nets += [copy.deepcopy(net) for net in nets]
    agents = [Agent(net, distance_f=distance_f, lr=lr) for net in nets]
    return agents, nets