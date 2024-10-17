import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import matplotlib.pyplot as plt
from Environment import SumoIntersectionEnv

class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.metrics_history = {
            'episode_reward': [],
            'episode_length': [],
            'total_waiting_time': [],
            'total_queue_length': [],
            'total_throughput': [],
            'average_speed': []
        }

    def _on_step(self) -> bool:
        if self.locals.get('dones')[0]:
            info = self.locals.get('infos')[0]
            if info:
                self.metrics_history['episode_reward'].append(info.get('cumulative_reward', 0))
                self.metrics_history['episode_length'].append(self.n_calls)
                self.metrics_history['total_waiting_time'].append(info.get('waiting_time', 0))
                self.metrics_history['total_queue_length'].append(info.get('queue_length', 0))
                self.metrics_history['total_throughput'].append(info.get('throughput', 0))
                self.metrics_history['average_speed'].append(info.get('average_speed', 0))
                
                # Log to tensorboard
                self.logger.record('metrics/episode_reward', info.get('cumulative_reward', 0))
                self.logger.record('metrics/episode_length', self.n_calls)
                self.logger.record('metrics/waiting_time', info.get('waiting_time', 0))
                self.logger.record('metrics/queue_length', info.get('queue_length', 0))
                self.logger.record('metrics/throughput', info.get('throughput', 0))
                self.logger.record('metrics/average_speed', info.get('average_speed', 0))
        return True

def plot_metrics(callback: TensorboardCallback):
    metrics = callback.metrics_history
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))
    fig.suptitle('Training Metrics')
    
    # Plot episode rewards
    axs[0, 0].plot(metrics['episode_reward'])
    axs[0, 0].set_title('Episode Rewards')
    axs[0, 0].set_xlabel('Episode')
    axs[0, 0].set_ylabel('Reward')
    
    # Plot average speed
    axs[0, 1].plot(metrics['average_speed'])
    axs[0, 1].set_title('Average Speed')
    axs[0, 1].set_xlabel('Episode')
    axs[0, 1].set_ylabel('Speed (m/s)')
    
    # Plot waiting times
    axs[1, 0].plot(metrics['total_waiting_time'])
    axs[1, 0].set_title('Total Waiting Time')
    axs[1, 0].set_xlabel('Episode')
    axs[1, 0].set_ylabel('Time (s)')
    
    # Plot queue lengths
    axs[1, 1].plot(metrics['total_queue_length'])
    axs[1, 1].set_title('Total Queue Length')
    axs[1, 1].set_xlabel('Episode')
    axs[1, 1].set_ylabel('Vehicles')
    
    # Plot throughput
    axs[2, 0].plot(metrics['total_throughput'])
    axs[2, 0].set_title('Total Throughput')
    axs[2, 0].set_xlabel('Episode')
    axs[2, 0].set_ylabel('Vehicles')
    
    # Plot episode lengths
    axs[2, 1].plot(metrics['episode_length'])
    axs[2, 1].set_title('Episode Lengths')
    axs[2, 1].set_xlabel('Episode')
    axs[2, 1].set_ylabel('Steps')
    
    plt.tight_layout()
    plt.show()

def make_env():
    def _init():
        env = SumoIntersectionEnv(gui=True)
        return env
    return _init

def train_agent(env, total_timesteps=100000):
    model = DQN(
        "MlpPolicy", 
        env, 
        verbose=1,
        tensorboard_log="./tensorboard_logs/",
        learning_rate=3e-4,  # Slightly increased learning rate
        buffer_size=200000,  # Increased buffer size
        learning_starts=5000,  # More initial random actions
        batch_size=128,  # Increased batch size
        tau=0.1,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.2,  # Longer exploration
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        policy_kwargs=dict(
            net_arch=[256, 256]  # Larger network
        )
    )
    callback = TensorboardCallback()
    
    try:
        model.learn(total_timesteps=total_timesteps, callback=callback)
        model.save("dqn_traffic_control")
    except Exception as e:
        print(f"Training interrupted: {e}")
    
    return model, callback

def evaluate_agent(model, env, num_episodes=10):
    all_rewards = []
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        terminated = truncated = False
        
        while not (terminated or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
        
        all_rewards.append(episode_reward)
    
    return np.mean(all_rewards), np.std(all_rewards)

if __name__ == "__main__":
    try:
        # Create vectorized environment
        env = make_vec_env(make_env(), n_envs=1)
        
        # Train the agent
        model, callback = train_agent(env, total_timesteps=100000)
        
        # Plot training metrics
        plot_metrics(callback)
        
        # Evaluate the agent
        mean_reward, std_reward = evaluate_agent(model, env)
        print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Make sure to close the environment
        env.close()