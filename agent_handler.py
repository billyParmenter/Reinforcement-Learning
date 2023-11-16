class Agent_handler():
  def __init__(self, agent_params):
    self.num_episodes = agent_params["num_episodes"]
    self.max_steps = agent_params["max_steps"]
    self.notify_percent = agent_params["notify_percent"]
    self.update_rate = agent_params["update_rate"]

  def train_agent(self, agent, env):
    episode_steps = []
    episode_returns = []
    progress = 0
    progress_delta = 0

    print(f'\tEpisode 0/{self.num_episodes} 0%')

    for episode in range(self.num_episodes):
      steps = 0
      total_reward = 0
      state, _ = env.reset()
      action = agent.select_action(state)

      while True:
        next_state, reward, done, _, _ = env.step(action)
        steps += 1
        total_reward += reward
        next_action = agent.select_action(next_state)
        agent.update_q_values(state, action, reward, next_state, done)

        if done or steps >= self.max_steps - 1:
          episode_steps.append(steps)
          episode_returns.append(total_reward)
          break

        state = next_state
        action = next_action

      if episode % self.update_rate == 0:
        agent.target_q_network.set_weights(agent.q_network.get_weights())

      progress_delta += round(((episode + 1) / self.num_episodes) * 100) - progress
      progress = round(((episode + 1) / self.num_episodes) * 100)

      if progress_delta >= self.notify_percent:
        print(f'\tEpisode {episode + 1}/{self.num_episodes} {progress}%')
        progress_delta = round(((episode + 1) / self.num_episodes) * 100) - progress
      elif progress >= 100:
        print(f'\tEpisode {episode + 1}/{self.num_episodes} {progress}%')
        pass

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
  
    

        