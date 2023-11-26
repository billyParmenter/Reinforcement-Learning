from collections import deque
from assignment3_utils import *

class Agent_handler():
  def __init__(self, agent_params):
    self.notify_percent = agent_params["notify_percent"]
    self.num_episodes = agent_params["num_episodes"]
    self.max_steps = agent_params["max_steps"]
    self.skip = agent_params["skip"]
    self.crop = agent_params["crop"]
    self.checkpoint_interval = agent_params["checkpoint_interval"]

    self.progress_delta = 0
    self.progress = 0


  def update_progress(self, episode):
    self.progress_delta += round(((episode) / self.num_episodes) * 100) - self.progress
    self.progress = round(((episode) / self.num_episodes) * 100)

    if self.progress_delta >= self.notify_percent:
      print(f'\tEpisode {episode + 1}/{self.num_episodes} {self.progress}%')
      self.progress_delta = round(((episode + 1) / self.num_episodes) * 100) - self.progress
    elif episode >= self.num_episodes:
      print(f'\tEpisode {episode + 1}/{self.num_episodes} 100%')



  def checkpoint(self, episode, agent):
    if episode % self.checkpoint_interval == 0 and episode != 0 and self.last_save != None:
      self.last_save = agent.checkpoint()


  def train_agent(self, agent, env):
    episode_steps = []
    episode_rewards = []
    episode_dones = []
    self.progress = 0
    self.progress_delta = 0
    self.last_save = ''

    print(f'\tEpisode 0/{self.num_episodes} 0%')

    for episode in range(self.num_episodes):
      steps = 0
      total_reward = 0

      state_reset = env.reset()
      frame = process_frame(state_reset[0], self.crop)
      images = deque(maxlen=4)
      images.append(frame)

      for _ in range(self.skip):
        state_step = env.step(0)
        frame = process_frame(state_step[0], self.crop)
        images.append(frame)

      try:
        while True:
          state = images

          action = agent.select_action(images)
          next_frame, reward, done, _, _ = env.step(action)

          next_frame = process_frame(next_frame, self.crop)
          images.append(next_frame)
          next_state = images

          agent.update_q_values(state, action, reward, next_state, done)

          total_reward += reward
          steps += 1

          agent.update_target_network(episode)
            

          if done or steps >= self.max_steps:
            episode_steps.append(steps)
            episode_rewards.append(total_reward)
            break


      except Exception as e:
        print("\t!! CRASH !!")
        if self.last_save != '':
          agent = agent(None, self.last_save)
          episode = round(episode, -2)
        else:
          raise e

      self.checkpoint(episode, agent)
      self.update_progress(episode)
    
    print("\nDone training!\n\n")

    return episode_steps, episode_rewards

  def train(self, agents, env):
    results = {}
    count = 1

    for agent in agents:
      print(f'~~~ Training Agent: {agent.name} {count}/{len(agents)} ~~~')

      average_steps, average_rewards = self.train_agent(agent, env)

      results[agent.name] = {"steps": average_steps, "rewards": average_rewards}

      agent.save("final")

      count += 1

    return results
  
    

        