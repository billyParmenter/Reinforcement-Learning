import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from collections import deque
import json

class DQN_Agent():
  def __init__(self, agent_params, load_path=None):
    self.num_obs = agent_params["num_obs"]
    self.num_actions = agent_params["num_actions"]
    self.learning_rate = agent_params["learning_rate"]
    self.discount_factor = agent_params["discount_factor"]
    self.exploration_factor = agent_params["exploration_factor"]
    self.verbose = 0

    buffer_size = 10000
    batch_size=32

    self.replay_buffer = deque(maxlen=buffer_size)
    self.batch_size = batch_size

    if load_path:
      self.load(load_path)

    else:
      self.q_network = self.build_q_network()
      self.target_q_network = self.build_q_network()


  def save(self, save_path):
    # Save agent_params
    with open(save_path + "_params.json", "w") as params_file:
      json.dump({
        "num_obs": self.num_obs,
        "num_actions": self.num_actions,
        "learning_rate": self.learning_rate,
        "discount_factor": self.discount_factor,
        "exploration_factor": self.exploration_factor
      }, params_file)

    # Save model weights
    self.q_network.save_weights(save_path + "_model.h5")

  def load(self, load_path):
    # Load agent_params
    with open(load_path + "_params.json", "r") as params_file:
      params_data = json.load(params_file)
      self.num_obs = params_data["num_obs"]
      self.num_actions = params_data["num_actions"]
      self.learning_rate = params_data["learning_rate"]
      self.discount_factor = params_data["discount_factor"]
      self.exploration_factor = params_data["exploration_factor"]

    # Build the model and load weights
    self.q_network = self.build_q_network()
    self.target_q_network = self.build_q_network()
    self.q_network.load_weights(load_path + "_model.h5")



  def sample_from_replay_buffer(self):
    batch = random.sample(self.replay_buffer, self.batch_size)

    batch = [list(entry) for entry in batch]

    max_length = max(len(entry) for entry in batch)
    batch = [entry + [None] * (max_length - len(entry)) for entry in batch]

    states, actions, rewards, next_states, dones = np.array(batch).T

    states = np.array(states.tolist(), dtype=np.uint8)
    next_states = np.array(next_states.tolist(), dtype=np.uint8)

    next_states = np.array([state.reshape((210, 160, 3)) for state in next_states])

    return states, actions, rewards, next_states, dones


  def build_q_network(self):
    model = tf.keras.Sequential([
      layers.Input(shape=self.num_obs),
      layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu'),
      layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu'),
      layers.Conv2D(64, (3, 3), activation='relu'),
      layers.Flatten(),
      layers.Dense(512, activation='relu'),
      layers.Dense(self.num_actions, activation='linear')  
    ])

    model.compile(
      loss=tf.keras.losses.Huber(),
      optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
      metrics=['accuracy']
    )

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

    if len(self.replay_buffer) >= self.batch_size:
      states, actions, rewards, next_states, dones = self.sample_from_replay_buffer()

      targets = rewards + self.discount_factor * np.max(self.target_q_network.predict(next_states, verbose=0), axis=1) * (1 - dones)

      target_q_values = self.q_network.predict(states, verbose=0)
      
      actions = actions.astype(int)
      
      target_q_values[np.arange(self.batch_size), actions] = (1 - self.learning_rate) * target_q_values[np.arange(self.batch_size), actions] + self.learning_rate * targets

      self.q_network.fit(states, target_q_values, epochs=1, verbose=self.verbose)


