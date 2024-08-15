import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import os
import datetime

class DQNAgent:
    def __init__(self, state_size, action_size):
        # Initialize the DQN agent with the size of the state and action space
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []  # Memory buffer for storing experiences
        self.gamma = 0.99  # Discount rate for future rewards
        self.epsilon = 1.0  # Initial exploration rate (probability of choosing a random action)
        self.epsilon_min = 0.01  # Minimum value for epsilon (ensures some exploration continues)
        self.epsilon_decay = 0.995  # Decay rate for epsilon (reduces exploration over time)
        self.learning_rate = 0.001  # Learning rate for the neural network optimizer
        self.model = self.build_model()  # Build the primary Q-network
        self.target_model = self.build_model()  # Build the target Q-network
        self.update_target_model()  # Initialize target model with weights from the primary model

        # Setup TensorBoard for logging training metrics
        log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.tensorboard_writer = tf.summary.create_file_writer(log_dir)

    def build_model(self):
        # Build a neural network model with three hidden layers and dropout for regularization
        model = Sequential()
        model.add(Dense(128, input_dim=self.state_size, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))  # Output layer with no activation (linear output for Q-values)
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))  # Compile the model using mean squared error loss and Adam optimizer
        return model

    def update_target_model(self):
        # Copy the weights from the primary Q-network to the target Q-network
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        # Store the experience tuple (state, action, reward, next_state, done) in memory
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # Decide whether to explore or exploit
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # Choose a random action (exploration)
        act_values = self.model.predict(state, verbose=0)  # Predict Q-values for the current state
        return np.argmax(act_values[0])  # Choose the action with the highest Q-value (exploitation)

    def replay(self, batch_size):
        # Train the model using a random sample of experiences from memory
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.target_model.predict(next_state, verbose=0)[0])  # Calculate the target Q-value
            target_f = self.model.predict(state, verbose=0)  # Get the predicted Q-values for the current state
            target_f[0][action] = target  # Update the Q-value for the action taken
            self.model.fit(state, target_f, epochs=1, verbose=0)  # Train the model with the updated Q-value
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay  # Decay epsilon to reduce the exploration rate over time

    def train(self, batch_size):
        # Train the agent if enough experiences are stored in memory
        if len(self.memory) > batch_size:
            self.replay(batch_size)  # Sample a batch and update the Q-network
            self.update_target_model()  # Periodically update the target Q-network with the weights from the primary Q-network
    
    def log_metrics(self, episode, score, avg_10_games, total_reward, avg_cumulative_reward):
        # Log training metrics to TensorBoard for visualization
        with self.tensorboard_writer.as_default():
            tf.summary.scalar('game_score', score, step=episode)
            tf.summary.scalar('avg_10_games', avg_10_games, step=episode)
            tf.summary.scalar('total_reward', total_reward, step=episode)
            tf.summary.scalar('avg_cumulative_reward', avg_cumulative_reward, step=episode)
            tf.summary.scalar('epsilon', self.epsilon, step=episode)
            self.tensorboard_writer.flush()  # Ensure that the metrics are written to the log
