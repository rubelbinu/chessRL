import numpy as np
from chess_env import ChessEnv
from dqn_agent import DQNAgent
import os
import logging
import warnings

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppresses detailed TensorFlow logging to reduce console clutter
logging.getLogger('tensorflow').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=FutureWarning)  # Suppresses future warnings
warnings.filterwarnings('ignore', category=UserWarning)  # Suppresses user warnings

def train():
    # Initialize the chess environment and DQN agent
    env = ChessEnv()
    state_size = len(env.get_state())  # Determine the size of the state space based on the environment's state
    action_size = len(list(env.board.legal_moves))  # Determine the size of the action space based on legal moves
    agent = DQNAgent(state_size, action_size)  # Instantiate the DQN agent with the state and action sizes
    episodes = 1000  # Define the number of training episodes
    batch_size = 64  # Define the batch size for training the DQN
    scores = []  # List to store the scores (total reward) for each episode
    total_rewards = []  # List to store cumulative rewards across episodes

    for e in range(episodes):
        state = env.reset()  # Reset the environment to its initial state at the start of each episode
        state = np.reshape(state, [1, state_size])  # Reshape the state to match the input shape expected by the model
        episode_reward = 0  # Initialize the total reward for the current episode

        for time in range(300):  # Set a maximum number of steps per episode (300)
            action = agent.act(state)  # Select an action using the agent's policy (either exploration or exploitation)
            next_state, reward, done = env.step(action)  # Execute the action in the environment and observe the outcome
            episode_reward += reward  # Accumulate the reward for the current step into the episode's total reward
            next_state = np.reshape(next_state, [1, state_size])  # Reshape the next state
            agent.remember(state, action, reward, next_state, done)  # Store the experience in the agent's memory
            state = next_state  # Update the current state to the next state

            if done:  # If the episode is over (checkmate, stalemate, etc.), exit the loop
                break
            
            # Render the environment during training (visualize the board and moves)
            env.render()
        
        # Train the DQN agent using the experiences stored in memory
        agent.train(batch_size)
        scores.append(episode_reward)  # Append the total reward of the episode to the scores list
        total_rewards.append(episode_reward)  # Append the episode reward to the total rewards list

        # Calculate the average cumulative reward of the last 5 episodes for monitoring progress
        avg_cumulative_reward = np.mean(total_rewards[-5:]) if len(total_rewards) >= 5 else np.mean(total_rewards)
        # Calculate the average reward over the last 10 games
        avg_10_games = np.mean(scores[-10:]) if len(scores) >= 10 else np.mean(scores)

        # Log metrics to TensorBoard for visualization
        agent.log_metrics(e, episode_reward, avg_10_games, episode_reward, avg_cumulative_reward)

        # Print episode statistics to the console for real-time monitoring
        print(f"Episode: {e}/{episodes}, score: {time}, total reward: {episode_reward:.2f}, epsilon: {agent.epsilon:.2f}")

if __name__ == "__main__":
    train()  # Start the training process when the script is run
