import torch
import os
import json
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from datetime import datetime
import random

class DQN_PyTorch(nn.Module):
    def __init__(self, num_obs, num_actions):
        super(DQN_PyTorch, self).__init__()
        self.num_actions = num_actions

        # Adjusted input size for 4 frames with height=81 and width=70
        self.features = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # Calculate the expected size for the fully connected layer
        expected_size = self.calculate_conv_output_size((4, 81, 70))

        self.fc = nn.Sequential(
            nn.Linear(2304, 1920),
            nn.ReLU(),
            nn.Linear(1920, self.num_actions)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        return x

    def calculate_conv_output_size(self, input_size):
        # Calculate output size for the first convolutional layer
        h_out = ((input_size[1] - 8) // 4) + 1
        w_out = ((input_size[2] - 8) // 4) + 1
        # Calculate output size for the second convolutional layer
        h_out = ((h_out - 4) // 2) + 1
        w_out = ((w_out - 4) // 2) + 1
        # Calculate output size for the third convolutional layer
        h_out = ((h_out - 3) // 1) + 1
        w_out = ((w_out - 3) // 1) + 1
        # Return the total number of features
        return h_out * w_out * 64


class DQN_Agent_PyTorch():
    def __init__(self, agent_params=None, file=None, use_cuda=True):
        self.verbose = 0
        self.replay_buffer = deque(maxlen=5000)
        self.use_cuda = use_cuda

        if use_cuda and torch.cuda.is_available():
            print("Using CUDA")
            self.device = torch.device("cuda")
        else:
            print("Using CPU")
            self.device = torch.device("cpu")

        if file:
            self.load(file)
        elif agent_params:
            self.init_params(agent_params)
        else:
            raise Exception("No params")

    def init_params(self, params):
        self.name = params["name"]
        self.num_obs = params["num_obs"]
        self.batch_size = params["batch_size"]
        self.update_rate = params["update_rate"]
        self.num_actions = params["num_actions"]
        self.gradient_clip = params["gradient_clip"]
        self.learning_rate = params["learning_rate"]
        self.discount_factor = params["discount_factor"]
        self.exploration_decay = params["exploration_decay"]
        self.exploration_factor = params["exploration_factor"]
        self.min_exploration_rate = params["min_exploration_rate"]

        self.q_network = DQN_PyTorch(self.num_obs, self.num_actions).to(self.device)
        self.target_q_network = DQN_PyTorch(self.num_obs, self.num_actions).to(self.device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())


        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.mse_loss = nn.MSELoss()

    def checkpoint(self):
        print(f'\tCheckpoint: {datetime.now().strftime("%m-%d %H:%M")}')
        return self.save("checkpoint")

    def file_pathing(self, suffix=None):
        if suffix:
            return f'./models/{self.name}_{suffix}_'
        return f'./models/{self.name}_'

    def save(self, name):
        os.makedirs("./models", exist_ok=True)
        # Save agent_params
        with open(f'{self.file_pathing(name)}params.json', "w") as params_file:
            json.dump({
                "name": self.name,
                "num_obs": self.num_obs,
                "batch_size": self.batch_size,
                "update_rate": self.update_rate,
                "num_actions": self.num_actions,
                "gradient_clip": self.gradient_clip,
                "learning_rate": self.learning_rate,
                "discount_factor": self.discount_factor,
                "exploration_decay": self.exploration_decay,
                "exploration_factor": self.exploration_factor,
                "min_exploration_rate": self.min_exploration_rate,
            }, params_file)

        # Save model weights
        torch.save(self.q_network.state_dict(), f'{self.file_pathing(name)}model.pth')

        return self.file_pathing(name)

    def load(self, file):
        # Load agent_params
        with open(f'./models/{file}_params.json', "r") as params_file:
            params_data = json.load(params_file)
            self.init_params(params_data)

        self.q_network.load_state_dict(torch.load(f'./models/{file}_model.pth', map_location=self.device))
        self.target_q_network.load_state_dict(self.q_network.state_dict())

    def update_target_network(self, episode):
        if episode % self.update_rate == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())

    def sample_from_replay_buffer(self):
        self.exploration_factor = max(self.min_exploration_rate, self.exploration_factor * self.exploration_decay)

        batch = random.sample(self.replay_buffer, self.batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.from_numpy(np.stack(states)).float().to(self.device)
        actions = torch.from_numpy(np.array(actions)).long().to(self.device)
        rewards = torch.from_numpy(np.array(rewards)).float().to(self.device)
        next_states = torch.from_numpy(np.stack(next_states)).float().to(self.device)
        dones = torch.from_numpy(np.array(dones)).float().to(self.device)

        return states, actions, rewards, next_states, dones

    def select_action(self, state):
        if np.random.rand() < self.exploration_factor:
            return np.random.randint(self.num_actions)
        else:
            return self.select_greedy_action(state)

    def select_greedy_action(self, state):
        # Convert the deque to a NumPy array
        state_array = np.array(state)
        
        # Convert the NumPy array to a PyTorch tensor
        state_tensor = torch.from_numpy(state_array).unsqueeze(0).float().to(self.device)

        # Get the Q values from the neural network
        q_values_state = self.q_network(state_tensor).detach().cpu().numpy()

        # Return the index of the action with the highest Q value
        return np.argmax(q_values_state)

    def update_q_values(self, state, action, reward, next_state, done):
        reward = np.clip(reward, -1, 1)

        self.replay_buffer.append((state, action, reward, next_state, done))

        if done and len(self.replay_buffer) >= self.batch_size:
            states, actions, rewards, next_states, dones = self.sample_from_replay_buffer()

            self.optimizer.zero_grad()

            q_states = self.q_network(states)
            q_next_states = self.target_q_network(next_states)

            targets = rewards + self.discount_factor * torch.max(q_next_states, dim=1)[0] * (1 - dones)
            predictions = q_states.gather(1, actions.unsqueeze(1))

            loss = self.mse_loss(predictions, targets.unsqueeze(1))
            loss.backward()

            # Gradient clipping
            if self.gradient_clip:
                torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.gradient_clip)

            self.optimizer.step()
