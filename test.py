import utils as Utils
from dqn_agent import DQN_Agent
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
    "learning_rate": 0.0001,
    "discount_factor": 0.95,
    "exploration_factor": 1,
    "min_exploration_rate": 0.05,
    "exploration_decay": 0.995,
    "batch_size": batch,
    "name": f'{batch}_{update}'
  }
  return params

agents = []

batchs = [8, 16]
updates = [3, 10]

for batch in batchs:
  for update in updates:
    agents.append(DQN_Agent(get_params(batch, update)))


handler = Agent_handler({
  "num_episodes":10,
  "max_steps":100,
  "notify_percent":1,
  "skip": 85,
  "checkpoint_interval": 200,
  "crop": {
    "top": 0,
    "bottom": -39,
    "left": 0,
    "right": -1,
  }
})

results = handler.train([agents[0]], env)

output_file_path = "results.json"
with open(output_file_path, "w") as json_file:
  json.dump(results, json_file)

print(f"Results saved to {output_file_path}")