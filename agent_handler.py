class Agent_handler():
  def __init__(self, agent_params):
    self.notify_percent = agent_params["notify_percent"]
    self.num_episodes = agent_params["num_episodes"]
    self.max_steps = agent_params["max_steps"]
    self.skip = agent_params["skip"]

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
    if episode % 100 == 0 and episode != 0 and self.last_save != None:
      self.last_save = agent.checkpoint()


  def train_agent(self, agent, env):
    episode_steps = []
    episode_rewards = []
    self.progress = 0
    self.progress_delta = 0
    self.last_save = ''

    print(f'\tEpisode 0/{self.num_episodes} 0%')

    for episode in range(self.num_episodes):
      steps = 0
      total_reward = 0

      state_frame, _ = env.reset()
      state = self.crop(state_frame)
      action = agent.select_action(state)

      for _ in range(self.skip):
        env.step(0)

      try:
        while True:
          frame, reward, done, _, _ = env.step(action)

          next_state = self.crop(frame)
          next_action = agent.select_action(next_state)
          agent.update_q_values(state, action, reward, next_state, done)

          total_reward += reward
          state = next_state
          action = next_action
          steps += 1

          agent.update_target_network(episode)
            
          self.checkpoint(episode, agent)

          if done or steps >= self.max_steps:
            episode_steps.append(steps)
            episode_rewards.append(total_reward)
            break


      except Exception as e:
        if self.last_save != '':
          agent = agent(None, self.last_save)
          episode = round(episode, -2)
        else:
          raise e

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

  def crop(self, frame):
    vertical_crop_start   = 0
    vertical_crop_end     = 171
    horizontal_crop_start = 0
    horizontal_crop_end   = 160
    cropped_frame = frame[vertical_crop_start:vertical_crop_end, horizontal_crop_start:horizontal_crop_end]
    return cropped_frame
  
    

        