import utils as Utils
from dqn_agent import DQN_Agent
from double_dqn_agent import Double_DQN_Agent
from dueling_dqn_agent import DuelingDQN_Agent
from agent_handler import Agent_handler
from assignment3_utils import *
import numpy as np
import gym
import tensorflow as tf
import json

gpu_available = tf.config.list_physical_devices('GPU')

if gpu_available:
  print("GPU is available.")
else:
  print("No GPU found. TensorFlow is using CPU.")

env = gym.make('MsPacman-v4', render_mode='rgb_array')

num_obs, num_actions = Utils.describe_env(env)

def get_params(batch, update):
  params = {
    "num_obs": (4, 86, 80),
    "num_actions": num_actions,
    "update_rate": update,
    "learning_rate": 0.001,
    "discount_factor": 0.95,
    "exploration_factor": 1,
    "min_exploration_rate": 0.05,
    "exploration_decay": 0.995,
    "batch_size": batch,
  }
  return params

agents = []

agents.append(DQN_Agent(get_params(16, 20)))
agents.append(Double_DQN_Agent(get_params(16, 20)))
agents.append(DuelingDQN_Agent(get_params(16, 20)))


handler = Agent_handler({
  "num_episodes":200,
  "max_steps":100_000,
  "notify_percent":10,
  "skip": 85,
  "checkpoint_interval": 50,
  "crop": {
    "top": 0,
    "bottom": -39,
    "left": 0,
    "right": -1,
  }
})

results = handler.train(agents, env)

output_file_path = "results.json"
with open(output_file_path, "w") as json_file:
  json.dump(results, json_file)

print(f"Results saved to {output_file_path}")