import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def describe_env(env):
    num_actions = env.action_space.n
    obs = env.observation_space
    num_obs = env.observation_space.shape

    print("Observation space: ", obs)
    print("Observation space size: ", num_obs)
    print("Number of actions: ", num_actions)

    return num_obs, num_actions


def plot_results(result, agent_name, smoothing=1):
    i = 1
    plt.figure(figsize=(12, 6))
    
    for metric, values in result.items():
        moving_avg_steps = np.convolve(values, np.ones(smoothing)/smoothing, mode='valid')

        plt.subplot(1, 3, i)
        plt.plot(moving_avg_steps)
        plt.title(f'{agent_name} - Episode {metric}')
        plt.xlabel('Episode')
        plt.ylabel(f'Smoothed {metric} (smoothing={smoothing})')

        i += 1

    plt.tight_layout()
    plt.show()


def crop(env):
    vertical_crop_start   = 0       # @param{type:"integer"}
    vertical_crop_end     = 171       # @param{type:"integer"}
    horizontal_crop_start = 0         # @param{type:"integer"}
    horizontal_crop_end   = 160       # @param{type:"integer"}

    env.reset()
    obs = env.render()

    cropped_obs = obs[vertical_crop_start:vertical_crop_end, horizontal_crop_start:horizontal_crop_end].shape
    cropped_obs