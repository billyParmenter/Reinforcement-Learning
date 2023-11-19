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


def plot_results(results, result_index, window_size, title):
    plt.figure(figsize=(25, 10))

    for result in results:
        sns.lineplot(
            np.convolve(
                result[result_index], np.ones(window_size) / window_size, mode="same"
            ),
            label=f"(LR, EF) {(result[3]) }",
        )

    plt.title(title)
    plt.legend()
    plt.show()
