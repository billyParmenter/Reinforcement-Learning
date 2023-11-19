import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from collections import deque
import json
from dqn_agent import DQN_Agent
class Double_DQN_Agent(DQN_Agent):
  def __init__(self, agent_params, load_path=None):
      super().__init__(agent_params, load_path)

  def update_q_values(self, state, action, reward, next_state, done):
      reward = np.clip(reward, -1, 1)
      self.replay_buffer.append((state, action, reward, next_state, done))

      if done and len(self.replay_buffer) >= self.batch_size:
          states, actions, rewards, next_states, dones = self.sample_from_replay_buffer()

          next_actions_online = np.argmax(self.q_network.predict(next_states, verbose=0), axis=1)

          targets = rewards + self.discount_factor * self.target_q_network.predict(next_states, verbose=0)[range(self.batch_size), next_actions_online] * (1 - dones)

          target_q_values = self.q_network.predict(states, verbose=0)
          actions = actions.astype(int)
          target_q_values[range(self.batch_size), actions] = (1 - self.learning_rate) * target_q_values[range(self.batch_size), actions] + self.learning_rate * targets

          self.q_network.fit(states, target_q_values, epochs=1, verbose=self.verbose)

