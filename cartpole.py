import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from torch.autograd import Variable


class Model(nn.Module):
    def __init__(self, input_size, action_space):
        super(Model, self).__init__()
        # define layers
        self.l1 = nn.Linear(input_size, 32)
        self.l2 = nn.Linear(32, 64)
        self.l3 = nn.Linear(64, action_space)

    def forward(self, x):
        # get q values
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        q_values = self.l3(x)
        return q_values


class ReplayMemory:
    def __init__(self, mem_size):
        super().__init__()
        self.memory = []
        self.capacity = mem_size

    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)

    def push(self, itm):
        self.memory.append(itm)
        if len(self.memory) > self.capacity:
            del self.memory[0]


class Brain:
    def __init__(self, input_size, action_space_size, gamma):
        super().__init__()
        self.model = Model(input_size, action_space_size)
        self.memory = ReplayMemory(10000)
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.gamma = gamma
        self.reward_window = []

    def select_action(self, new_state):
        probs = F.softmax(self.model(new_state.detach()) * 70)  # temp=100
        actions = probs.multinomial(1)
        return actions.data[0, 0]

    def learn(self, batch_last_state, batch_new_state, batch_last_action, batch_last_reward):
        # todo check whats happening here
        # gets max values in self.model(batch_last_state) because batch_last_action contains index of value
        outputs = self.model(batch_last_state).gather(1, batch_last_action.unsqueeze(1)).squeeze(1)

        # print(self.model(batch_last_state),batch_last_action,outputs)
        next_outputs = self.model(batch_new_state).detach().max(1)[0]
        # print(next_outputs)
        # gamma * maxQValueOfNext + lastStateReward
        target = self.gamma * next_outputs + batch_last_reward
        # optimize the weights
        # td_loss = F.smooth_l1_loss(outputs, target)
        # print(outputs, target)
        td_loss = F.mse_loss(outputs, target)
        self.optimizer.zero_grad()
        td_loss.backward(retain_graph=True)
        self.optimizer.step()

    def update(self, reward, signal):
        new_state = torch.Tensor(signal).float().unsqueeze(0)
        # print(new_state) -> tensor([[ 0.0000,  0.0000,  0.0000,  0.6541, -0.6541]])
        self.memory.push(
            (self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward]))
        )
        # print(self.memory.memory)
        new_action = self.select_action(new_state)
        # learn
        if len(self.memory.memory) > 100:
            batch_last_state, batch_new_state, batch_last_action, batch_last_reward = self.memory.sample(100)
            self.learn(batch_last_state, batch_new_state, batch_last_action, batch_last_reward)

        self.last_action = new_action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return new_action

    def score(self):
        return sum(self.reward_window) / (len(self.reward_window) + 1.0)


env = gym.make("CartPole-v0")
env.reset()
observation, reward, done, _ = env.step(env.action_space.sample())
brain = Brain(len(observation), env.action_space.n, 0.001)
env.close()

for i in range(1000):
    env.reset()
    # if i % 20 == 0:
    # env.render()
    for j in range(1000):
        action = brain.update(reward, observation).numpy()
        # print(action)
        observation2, reward2, done, _ = env.step(action)  # take a random action
        if not done:
            observation = observation2
            reward = reward2
        else:
            print(j)
            break
    env.close()
    # print(brain.score())

# print(len(observation), env.action_space.n)

