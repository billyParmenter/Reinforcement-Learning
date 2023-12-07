import tensorflow as tf
from tensorflow.keras import layers
from dqn_agent import DQN_Agent


class DuelingDQN_Agent(DQN_Agent):
    def __init__(self, agent_params=None, load_path=None):
      super().__init__(agent_params, load_path)
      self.name = "Dueling_DQN"

    def build_q_network(self):
        input_layer = layers.Input(shape=self.num_obs)

        permuted_input = layers.Permute((2, 3, 1))(input_layer)

        conv1 = layers.Conv2D(32, (8, 8), strides=(4, 4), activation="relu")(
            permuted_input
        )
        conv2 = layers.Conv2D(64, (4, 4), strides=(2, 2), activation="relu")(conv1)
        conv3 = layers.Conv2D(64, (3, 3), activation="relu")(conv2)
        flatten = layers.Flatten()(conv3)

        # Value stream
        value_stream = layers.Dense(512, activation="relu")(flatten)
        value = layers.Dense(1, activation="linear")(value_stream)

        # Advantage stream
        advantage_stream = layers.Dense(512, activation="relu")(flatten)
        advantage = layers.Dense(self.num_actions, activation="linear")(
            advantage_stream
        )

        # Combine value and advantage to get Q values
        q_values = value + (
            advantage - tf.reduce_mean(advantage, axis=1, keepdims=True)
        )

        model = tf.keras.Model(inputs=input_layer, outputs=q_values)

        model.compile(
            loss=tf.keras.losses.Huber(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            metrics=["accuracy"],
        )

        return model
