class Agent_handler():
  def __init__(self, agent_params):
    self.num_episodes = agent_params["num_episodes"]
    self.max_steps = agent_params["max_steps"]
    self.notify_percent = agent_params["notify_percent"]
    self.update_rate = agent_params["update_rate"]
    self.skip = agent_params["skip"]

  def train_agent(self, agent, env):
    episode_steps = []
    episode_returns = []
    progress = 0
    progress_delta = 0

    print(f'\tEpisode 0/{self.num_episodes} 0%')

    for episode in range(self.num_episodes):
      steps = 0
      total_reward = 0
      state_frame, _ = env.reset()
      state = self.crop(state_frame)
      action = agent.select_action(state)
      last_save = ''
      training = True

      for _ in range(self.skip): # skip the start of each game
        env.step(0)

      while training:
        try:
          frame, reward, done, _, _ = env.step(action)
          next_state = self.crop(frame)
          steps += 1
          total_reward += reward
          next_action = agent.select_action(next_state)
          agent.update_q_values(state, action, reward, next_state, done)

          if done or steps >= self.max_steps:
            episode_steps.append(steps)
            episode_returns.append(total_reward)
            training = False
            break

          state = next_state
          action = next_action

          if episode % self.update_rate == 0:
            agent.target_q_network.set_weights(agent.q_network.get_weights())

          if episode % 100 == 0 and episode != 0:
            last_save = agent.checkpoint()

          progress_delta += round(((episode + 1) / self.num_episodes) * 100) - progress
          progress = round(((episode + 1) / self.num_episodes) * 100)

          if progress_delta >= self.notify_percent:
            print(f'\tEpisode {episode + 1}/{self.num_episodes} {progress}%')
            progress_delta = round(((episode + 1) / self.num_episodes) * 100) - progress
          elif progress >= 100:
            print(f'\tEpisode {episode + 1}/{self.num_episodes} {progress}%')

        except Exception as e:
          if last_save != '':
            agent = agent(None, last_save)
          else:
            raise e

      print("\nDone training!\n\n")

    return episode + 1, episode_steps, episode_returns

  def train(self, agents, env):
    results = []
    count = 1

    for agent in agents:
      print(f'~~~ Training Agent {count} {count}/{len(agents)} ~~~')

      num_episodes, average_steps, average_return = self.train_agent(agent, env)

      results.append((num_episodes, average_steps, average_return, f'Agent #{count}'))

    return results

  def crop(self, frame):
    vertical_crop_start   = 0
    vertical_crop_end     = 171
    horizontal_crop_start = 0
    horizontal_crop_end   = 160
    cropped_frame = frame[vertical_crop_start:vertical_crop_end, horizontal_crop_start:horizontal_crop_end]
    return cropped_frame
  
    

        