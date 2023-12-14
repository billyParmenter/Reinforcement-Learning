import matplotlib.pyplot as plt
import numpy as np

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

def process_frame(img, crop, verbose=False):

  img_cropped = img_crop(img, crop)

  img_downsampled = downsample(img_cropped)  # Crop and downsize (by 2)

  img_grayscale = to_grayscale(img_downsampled)  # Convert to greyscale by averaging the RGB values

  img_normalized = normalize_grayscale(img_grayscale)  # Normalize from -1 to 1.

  if verbose:
    print("Original Image Shape:", img.shape)
    plt.imshow(img)
    plt.title("Original Image")
    plt.show()
    print("Cropped Image Shape:", img_cropped.shape)
    plt.imshow(img_cropped)
    plt.title("Cropped Image")
    plt.show()
    print("Downsampled Image Shape:", img_downsampled.shape)
    plt.imshow(img_downsampled)
    plt.title("Downsampled Image")
    plt.show()
    print("Grayscale Image Shape:", img_grayscale.shape)
    plt.imshow(img_grayscale, cmap='gray')
    plt.title("Grayscale Image")
    plt.show()
    print("Normalized Image Shape:", img_normalized.shape)
    plt.imshow(img_normalized, cmap='gray')
    plt.title("Normalized Image")
    plt.show()

  return img_normalized


# Prints some information about the environment and returns the number ov observations and actions
def describe_env(env):
  num_actions = env.action_space.n
  obs = env.observation_space
  num_obs = env.observation_space.shape

  print("Observation space: ", obs)
  print("Observation space size: ", num_obs)
  print("Number of actions: ", num_actions)

  return num_obs, num_actions


# Takes an agents results in the form of a dictionary to be plotted and the agents name to be added to the plot
# can be given a smoothing window size
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