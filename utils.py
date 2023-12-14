import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import numpy as np
import matplotlib.pyplot as plt
# This is specific to the pong environment
def img_crop(img, crop):
    return img[crop["top"]:crop["bottom"],crop["left"]:crop["right"],:]

# GENERAL Atari preprocessing steps
def downsample(img):
    # We will take only half of the image resolution
    return img[::2, ::2]

def transform_reward(reward):
    # return np.sign(reward)
    return reward

def to_grayscale(img):
    return np.mean(img, axis=2).astype(np.uint8)

# Normalize grayscale image from -1 to 1.
def normalize_grayscale(img):
    return (img - 128) / 128 - 1  

def process_frame(img, crop):
    # print("Original Image Shape:", img.shape)
    # plt.imshow(img)
    # plt.title("Original Image")
    # plt.show()

    img_cropped = img_crop(img, crop)
    # print("Cropped Image Shape:", img_cropped.shape)
    # plt.imshow(img_cropped)
    # plt.title("Cropped Image")
    # plt.show()

    img_downsampled = downsample(img_cropped)  # Crop and downsize (by 2)
    # print("Downsampled Image Shape:", img_downsampled.shape)
    # plt.imshow(img_downsampled)
    # plt.title("Downsampled Image")
    # plt.show()

    img_grayscale = to_grayscale(img_downsampled)  # Convert to greyscale by averaging the RGB values
    # print("Grayscale Image Shape:", img_grayscale.shape)
    # plt.imshow(img_grayscale, cmap='gray')
    # plt.title("Grayscale Image")
    # plt.show()

    img_normalized = normalize_grayscale(img_grayscale)  # Normalize from -1 to 1.
    # print("Normalized Image Shape:", img_normalized.shape)
    # plt.imshow(img_normalized, cmap='gray')
    # plt.title("Normalized Image")
    # plt.show()

    return img_normalized


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
    plt.figure(figsize=(20, 6))
    
    for metric, values in result.items():
        moving_avg_steps = np.convolve(values, np.ones(smoothing)/smoothing, mode='valid')

        plt.subplot(1, 3, i)
        plt.plot(moving_avg_steps)
        plt.title(f'{agent_name} - Episode {metric}')
        plt.xlabel('Episode')
        plt.ylabel(f'{metric}')

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