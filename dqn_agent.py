import os
import tensorflow as tf
import numpy as np
import random
import json

from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from collections import deque
from datetime import datetime
class DQN_Agent():
  def __init__(self, agent_params=None, file=None):
    self.verbose = 0

    self.replay_buffer = deque(maxlen=5000)

    if file:
      self.load(file)

    elif agent_params:
      self.init_params(agent_params)

    else:
      raise Exception("No params")


  def init_params(self, params):
    self.name = "DQN"
    self.num_obs = params["num_obs"]
    self.batch_size=params["batch_size"]
    self.update_rate = params["update_rate"]
    self.num_actions = params["num_actions"]
    self.learning_rate = params["learning_rate"]
    self.discount_factor = params["discount_factor"]
    self.exploration_decay = params["exploration_decay"]
    self.exploration_factor = params["exploration_factor"]
    self.min_exploration_rate = params["min_exploration_rate"]

    self.q_network = self.build_q_network()
    self.target_q_network = self.build_q_network()
    self.target_q_network.set_weights(self.q_network.get_weights())


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
        "learning_rate": self.learning_rate,
        "discount_factor": self.discount_factor,
        "exploration_decay": self.exploration_decay,
        "exploration_factor": self.exploration_factor,
        "min_exploration_rate": self.min_exploration_rate,
      }, params_file)

    # Save model weights
    self.q_network.save_weights(f'{self.file_pathing(name)}model.h5')

    return self.file_pathing(name)

  def load(self, file):
    # Load agent_params
    with open(f'./models/{file}_params.json', "r") as params_file:
      params_data = json.load(params_file)
      self.init_params(params_data)

    self.q_network.load_weights(f'./models/{file}_model.h5')


  
  def update_target_network(self, episode):
    if episode % self.update_rate == 0:
      self.target_q_network.set_weights(self.q_network.get_weights())



  def sample_from_replay_buffer(self):
    self.exploration_factor = max(self.min_exploration_rate, self.exploration_factor * self.exploration_decay)

    batch = random.sample(self.replay_buffer, self.batch_size)

    batch = [list(entry) for entry in batch]

    max_length = max(len(entry) for entry in batch)
    batch = [entry + [None] * (max_length - len(entry)) for entry in batch]

    states, actions, rewards, next_states, dones = np.array(batch, dtype=object).T



    states = np.array(states.tolist(), dtype=np.uint8)
    next_states = np.array(next_states.tolist(), dtype=np.uint8)

    next_states = np.array([state.reshape(self.num_obs) for state in next_states])

    return states, actions, rewards, next_states, dones


  def build_q_network(self):
    model = tf.keras.Sequential([
      layers.Input(shape=self.num_obs),
      layers.Permute((2, 3, 1)),
      layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu'),
      layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu'),
      layers.Conv2D(64, (3, 3), activation='relu'),
      layers.Flatten(),
      layers.Dense(512, activation='relu'),
      layers.Dense(self.num_actions, activation='linear')  
    ])


    model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')

    return model


  def select_action(self, state):
    if np.random.rand() < self.exploration_factor:
      return np.random.randint(self.num_actions)
    else:
      return self.select_greedy_action(state)


  def select_greedy_action(self, state):
    q_values_state = self.q_network.predict(np.expand_dims(state, axis=0), verbose=0)[0]
    return np.argmax(q_values_state)


  def update_q_values(self, state, action, reward, next_state, done):
    reward = np.clip(reward, -1, 1)

    self.replay_buffer.append((state, action, reward, next_state, done))

    if done and len(self.replay_buffer) >= self.batch_size:
      states, actions, rewards, next_states, dones = self.sample_from_replay_buffer()

      targets = rewards + self.discount_factor * np.max(self.target_q_network.predict(next_states, verbose=0), axis=1) * (1 - dones)

      target_q_values = self.q_network.predict(states, verbose=0)
      
      actions = actions.astype(int)

      target_q_values[np.arange(self.batch_size), actions] = (1 - self.learning_rate) * target_q_values[np.arange(self.batch_size), actions] + self.learning_rate * targets

      self.q_network.fit(states, target_q_values, epochs=1, verbose=self.verbose)

