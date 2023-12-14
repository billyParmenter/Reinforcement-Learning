import tensorflow as tf
import utils as Utils
import json
import gym

from dueling_dqn_agent import DuelingDQN_Agent
from double_dqn_agent import Double_DQN_Agent
from agent_handler import Agent_handler
from dqn_agent import DQN_Agent

#Check to see if there is a GPU available in the environment(Computer)
gpu_available = tf.config.list_physical_devices('GPU')

if gpu_available:
  print("GPU is available.")
else:
  print("No GPU found. TensorFlow is using CPU.")


# Initialize the environtment (Agent)
env = gym.make('MsPacman-v4', render_mode='rgb_array')

# Setup the agent parameters
num_obs, num_actions = Utils.describe_env(env)

params = {
  "num_obs": (4, 86, 80),
  "num_actions": num_actions,
  "update_rate": 20,
  "learning_rate": 0.00005,
  "discount_factor": 0.95,
  "exploration_factor": 1,
  "min_exploration_rate": 0.05,
  "exploration_decay": 0.995,
  "batch_size": 16,
}

agents = []

agents.append(DQN_Agent(params))
agents.append(Double_DQN_Agent(params))
agents.append(DuelingDQN_Agent(params))


# Setup the handler parameters
handler = Agent_handler({
  "num_episodes":1,
  "max_steps":2,
  "notify_percent":10,
  "skip": 85,
  "checkpoint_interval": 100,
  "crop": {
    "top": 0,
    "bottom": -39,
    "left": 0,
    "right": -1,
  }
})

results = handler.train(agents, env)

# Sometimes we are working in a container so visualization is not available
# We save it to a file and open it in the project.ipynb on a machine that has
# a display
output_file_path = "results2.json"
with open(output_file_path, "w") as json_file:
  json.dump(results, json_file)

print(f"Results saved to {output_file_path}")