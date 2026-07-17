import random
import numpy as np 
import collections
import torch 
import torch.nn as nn 
import torch.optim as optim 

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU,
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.fc(x) 

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.action_dim = action_dim
        self.q_network = QNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        self.memory = ReplayBuffer(10000)
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.gamma = 0.99
        self.target_update_freq = 100
        self.target_q_network = QNetwork(state_dim, action_dim)
        self.target_q_network.load_state_dict(self.q_network.state_dict())

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            q_values = self.q_network(torch.tensor(state, dtype=torch.float32))
            return torch.argmax(q_values).item()
    
    def train_step(self):
        if len(self.memory) < 32:
            return
        states, actions, rewards, next_states, dones = self.memory.sample(32)
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.bool)
        
        q_values = self.q_network(states)
        next_q_values = self.target_q_network(next_states)
        target_q_values = rewards + self.gamma * torch.max(next_q_values, dim=1)[0] * ~dones

        loss = nn.MSELoss()(q_values.gather(1, actions.unsqueeze(1)), target_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        
        if step % self.target_update_freq == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())

state_dim, action_dim = 4, 2
agent = DQNAgent(state_dim, action_dim)
target_update_freq = 10

for e in range(100):
    state = np.random.rand(state_dim)
    total_reward = 0
    for step in range(100):
        action = agent.choose_action(state)
        next_state = np.random.rand(state_dim)
        reward = np.random.rand()
        done = np.random.rand() < 0.1
        agent.memory.push(state, action, reward, next_state, done)
        agent.train_step()
        state = next_state
        total_reward += reward
        if done:
            break
    print(f"Episode {e}, Total Reward: {total_reward}")