import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


class Agent():
    def __init__(self, agent_params):
        self.num_obs = agent_params["num_obs"]
        self.num_actions = agent_params["num_actions"]
        self.learning_rate = agent_params["learning_rate"]
        self.discount_factor = agent_params["discount_factor"]
        self.exploration_factor = agent_params["exploration_factor"]
        self.num_episodes = agent_params["num_episodes"]
        self.max_steps = agent_params["max_steps"]
        self.verbose = agent_params["verbose"]
        self.notify_percent = agent_params["notify_percent"]

        self.q_network = self.build_q_network()

    def build_q_network(self):
        model = tf.keras.Sequential([
            layers.Input(shape=self.num_obs),
            layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu'),
            layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu'),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dense(self.num_actions, activation='linear')  # Output layer with one node for each action
        ])

        model.compile(
            loss=tf.keras.losses.Huber(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            metrics=['accuracy']
        )

        return model


    #---------------------------
    # Functions are altered from week 5 "TD Learning Algorithms Implementation"
    #---------------------------
    def select_action(self, state):
        # Epsilon-greedy policy
        if np.random.rand() < self.exploration_factor:
            return np.random.randint(self.num_actions)  # Random action with probability epsilon
        else:
            return self.select_greedy_action(state) # Greedy action with probability (1 - epsilon)

    def select_greedy_action(self, state):
        q_values_state = self.q_network.predict(np.expand_dims(state, axis=0), verbose=0)[0]
        return np.argmax(q_values_state)


    def update_q_values(self, state, action, reward, next_state):
        reward = np.clip(reward, -1, 1)

        target = reward + self.discount_factor * np.max(self.q_network.predict(np.expand_dims(next_state, axis=0), verbose=0)[0])
        target_q_values = self.q_network.predict(np.expand_dims(state, axis=0), verbose=0)[0]
        target_q_values[action] = (1 - self.learning_rate) * target_q_values[action] + self.learning_rate * target
        self.q_network.fit(np.expand_dims(state, axis=0), np.expand_dims(target_q_values, axis=0), epochs=1, verbose=self.verbose)

    
    # def print_progress(self):
    #     if progress_delta >= self.notify_percent or episode == 0:
    #         print(f'\tEpisode {episode + 1}/{self.num_episodes} {round( ((episode + 1) / self.num_episodes) * 100 )}%')
    #         progress_delta = round( ((episode + 1) / self.num_episodes) * 100 ) - progress
    #     else:
    #         progress_delta += round( ((episode + 1) / self.num_episodes) * 100 ) - progress

    #     progress = round( ((episode + 1)/ self.num_episodes) * 100 )


    #---------------------------
    # Function was modified from week 5s
    #---------------------------
    def train(self, env):
        episode_steps = []
        episode_returns = []
        progress = 0
        progress_delta = 0

        print("Started Training...\n")
        print(f'\tEpisode 0/{self.num_episodes} 0%')


        for episode in range(self.num_episodes):

            steps = 0
            total_reward = 0
            state, _ = env.reset()
            action = self.select_action(state)

            while True:
                next_state, reward, done, _, _ = env.step(action)
                steps += 1
                total_reward += reward
                next_action = self.select_action(next_state)
                self.update_q_values(state, action, reward, next_state)

                if done or steps >= self.max_steps - 1:
                    episode_steps.append(steps)
                    episode_returns.append(total_reward)
                    break

                state = next_state
                action = next_action
            
            progress_delta += round( ((episode + 1) / self.num_episodes) * 100 ) - progress
            progress = round( ((episode + 1) / self.num_episodes) * 100 )

            if progress_delta >= self.notify_percent:
                print(f'\tEpisode {episode + 1}/{self.num_episodes} {progress}%')
                progress_delta = round( ((episode + 1) / self.num_episodes) * 100 ) - progress
            elif progress >= 100:
                print(f'\tEpisode {episode + 1}/{self.num_episodes} {progress}%')
                pass


        print("\nDone training!")

        return episode+1, episode_steps, episode_returns



