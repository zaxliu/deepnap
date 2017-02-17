import sys
sys.path.append('../')

from collections import deque
from qtable import QAgent
from simple_envs import SimpleMaze
from qnn_theano import QAgentNN
from mixin import PhiMixin


class QAgentPhi(PhiMixin, QAgent):
    def __init__(self, **kwargs):
        super(QAgentPhi, self).__init__(**kwargs)


class QAgentNNPhi(PhiMixin, QAgentNN):
    def __init__(self, **kwargs):
        super(QAgentNNPhi, self).__init__(**kwargs)


# PhiMixin test
agent_type = 'QAgent'
maze = SimpleMaze()

phi_length = 2
if agent_type == 'QAgent':
    agent = QAgentPhi(
        phi_length=phi_length,
        actions=maze.ACTIONS, alpha=0.5, gamma=0.5, explore_strategy='epsilon', epsilon=0.1)
elif agent_type == 'QAgentNN':
    slice_range = [(0, 3), (0, 4)] + zip([0] * len(maze.ACTIONS), [1] * len(maze.ACTIONS))
    agent = QAgentNNPhi(
        phi_length=phi_length,
        dim_state=(1, phi_length, 2+len(maze.ACTIONS)),
        range_state=[[slice_range]*phi_length],
        actions=maze.ACTIONS,
        learning_rate=0.001, reward_scaling=100, batch_size=100,
        freeze_period=100, memory_size=1000,
        alpha=0.5, gamma=0.5, explore_strategy='epsilon', epsilon=0.1, verbose=2)
else:
    raise ValueError("Unrecognized agent type!")
print "Maze and agent initialized!"

# logging
path = deque()  # path in this episode
episode_reward_rates = []
num_episodes = 0
cum_reward = 0
cum_steps = 0

# repeatedly run episodes
while True:
    maze.reset()
    agent.reset()
    action, _ = agent.observe_and_act(observation=None, last_reward=None)  # get and random action
    path.clear()
    episode_reward = 0
    episode_steps = 0
    episode_loss = 0

    # interact and reinforce repeatedly
    while not maze.isfinished():
        new_observation, reward = maze.interact(action)
        action, loss = agent.observe_and_act(observation=new_observation, last_reward=reward)
        # print action,
        # print new_observation,
        path.append(new_observation)
        episode_reward += reward
        episode_steps += 1
        episode_loss += loss if loss else 0
    # print len(path),
    # print "{:.3f}".format(episode_loss),
    # print ""
    cum_steps += episode_steps
    cum_reward += episode_reward
    num_episodes += 1
    episode_reward_rates.append(episode_reward / episode_steps)
    if num_episodes % 1000 == 0:
        print ""
        print num_episodes, len(agent._QAgent__q_table), cum_reward, cum_steps, 1.0 * cum_reward / cum_steps, cum_steps / 1000.0 #, path
        cum_reward = 0
        cum_steps = 0


