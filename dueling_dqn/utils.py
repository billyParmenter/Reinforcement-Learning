import matplotlib.pyplot as plt
import numpy as np


def describe_env(env):
    num_actions = env.action_space.n
    obs = env.observation_space
    num_obs = env.observation_space.shape

    print("Observation space: ", obs)
    print("Observation space size: ", num_obs)
    print("Number of actions: ", num_actions)

    return num_obs, num_actions


def plot_results(results, agent_name, smoothing=1):
    episode_steps = [result[1] for result in results]
    episode_returns = [result[2] for result in results]

    flat_steps = [item for sublist in episode_steps for item in sublist]
    flat_returns = [item for sublist in episode_returns for item in sublist]

    moving_avg_steps = np.convolve(
        flat_steps, np.ones(smoothing) / smoothing, mode="valid"
    )
    moving_avg_returns = np.convolve(
        flat_returns, np.ones(smoothing) / smoothing, mode="valid"
    )

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(moving_avg_steps)
    plt.title(f"{agent_name} - Episode Steps")
    plt.xlabel("Step")
    plt.ylabel(f"Smoothed Steps (smoothing={smoothing})")

    plt.subplot(1, 2, 2)
    plt.plot(moving_avg_returns)
    plt.title(f"{agent_name} - Episode Returns")
    plt.xlabel("Step")
    plt.ylabel(f"Smoothed Returns (smoothing={smoothing})")

    plt.tight_layout()
    plt.show()
