import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.monitor import Monitor

from environment2 import SumoIntersectionEnv

class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.metrics_history = {
            'episode_reward': [],
            'episode_length': [],
            'waiting_time': [],
            'queue_length': [],
            'throughput': [],
            'average_speed': []
        }

    def _on_step(self) -> bool:
        if self.locals.get('dones')[0]:
            info = self.locals.get('infos')[0]
            if info:
                episode_reward = info.get('episode', {}).get('r', 0)
                episode_length = info.get('episode', {}).get('l', 0)
                
                moving_averages = info.get('moving_averages', {})
                current_metrics = info.get('current_metrics', {})
                
                self.metrics_history['episode_reward'].append(episode_reward)
                self.metrics_history['episode_length'].append(episode_length)
                self.metrics_history['waiting_time'].append(moving_averages.get('waiting_time', 0))
                self.metrics_history['queue_length'].append(moving_averages.get('queue_length', 0))
                self.metrics_history['throughput'].append(current_metrics.get('total_exits', 0))
                self.metrics_history['average_speed'].append(moving_averages.get('average_speed', 0))
                
                # Log to tensorboard
                self.logger.record('metrics/episode_reward', episode_reward)
                self.logger.record('metrics/episode_length', episode_length)
                self.logger.record('metrics/waiting_time', moving_averages.get('waiting_time', 0))
                self.logger.record('metrics/queue_length', moving_averages.get('queue_length', 0))
                self.logger.record('metrics/throughput', current_metrics.get('total_exits', 0))
                self.logger.record('metrics/average_speed', moving_averages.get('average_speed', 0))
        return True

def plot_metrics(callback: TensorboardCallback):
    metrics = callback.metrics_history
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))
    fig.suptitle('Training Metrics')
    
    episodes = range(len(metrics['episode_reward']))
    
    # Plot episode rewards
    axs[0, 0].plot(episodes, metrics['episode_reward'])
    axs[0, 0].set_title('Episode Rewards')
    axs[0, 0].set_xlabel('Episode')
    axs[0, 0].set_ylabel('Reward')
    
    # Plot average speed
    axs[0, 1].plot(episodes, metrics['average_speed'])
    axs[0, 1].set_title('Average Speed')
    axs[0, 1].set_xlabel('Episode')
    axs[0, 1].set_ylabel('Speed (m/s)')
    
    # Plot waiting times
    axs[1, 0].plot(episodes, metrics['waiting_time'])
    axs[1, 0].set_title('Average Waiting Time')
    axs[1, 0].set_xlabel('Episode')
    axs[1, 0].set_ylabel('Time (s)')
    
    # Plot queue lengths
    axs[1, 1].plot(episodes, metrics['queue_length'])
    axs[1, 1].set_title('Average Queue Length')
    axs[1, 1].set_xlabel('Episode')
    axs[1, 1].set_ylabel('Vehicles')
    
    # Plot throughput
    axs[2, 0].plot(episodes, metrics['throughput'])
    axs[2, 0].set_title('Total Throughput')
    axs[2, 0].set_xlabel('Episode')
    axs[2, 0].set_ylabel('Vehicles')
    
    # Plot episode lengths
    axs[2, 1].plot(episodes, metrics['episode_length'])
    axs[2, 1].set_title('Episode Lengths')
    axs[2, 1].set_xlabel('Episode')
    axs[2, 1].set_ylabel('Steps')
    
    plt.tight_layout()
    plt.show()

def make_env():
    def _init():
        env = SumoIntersectionEnv(gui=False)  # Set gui=True for visualization
        return Monitor(env)  # Wrap with Monitor for proper episode tracking
    return _init

def train_agent(env, total_timesteps=100000):
    model = DQN(
        "MlpPolicy", 
        env, 
        verbose=1,
        tensorboard_log="./tensorboard_logs/",
        learning_rate=1e-4,  # Slightly lower learning rate for stability
        buffer_size=100000,  # Buffer size appropriate for episode length
        learning_starts=10000,  # More initial exploration
        batch_size=256,  # Larger batch size for better gradients
        tau=0.005,  # Slower target network update
        gamma=0.99,  # Standard discount factor
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.3,  # Longer exploration
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        policy_kwargs=dict(
            net_arch=[256, 256]  # Two hidden layers
        )
    )
    callback = TensorboardCallback()
    
    try:
        model.learn(total_timesteps=total_timesteps, callback=callback)
        model.save("dqn_traffic_control")
    except Exception as e:
        print(f"Training interrupted: {e}")
    finally:
        env.close()
    
    return model, callback

def evaluate_agent(model, env, num_episodes=10):
    all_rewards = []
    all_throughputs = []
    all_waiting_times = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        terminated = truncated = False
        episode_throughput = 0
        episode_waiting_time = 0
        
        while not (terminated or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            # Track additional metrics
            current_metrics = info.get('current_metrics', {})
            episode_throughput = max(episode_throughput, current_metrics.get('total_exits', 0))
            episode_waiting_time = current_metrics.get('waiting_time', 0)
        
        all_rewards.append(episode_reward)
        all_throughputs.append(episode_throughput)
        all_waiting_times.append(episode_waiting_time)
    
    return {
        'reward': (np.mean(all_rewards), np.std(all_rewards)),
        'throughput': (np.mean(all_throughputs), np.std(all_throughputs)),
        'waiting_time': (np.mean(all_waiting_times), np.std(all_waiting_times))
    }

if __name__ == "__main__":
    try:
        # Create vectorized environment
        env = make_vec_env(make_env(), n_envs=1)
        
        # Train the agent
        model, callback = train_agent(env, total_timesteps=100000)
        
        # Plot training metrics
        plot_metrics(callback)
        
        # Evaluate the agent
        eval_env = make_vec_env(make_env(), n_envs=1)
        eval_results = evaluate_agent(model, eval_env)
        
        # Print evaluation results
        print("\nEvaluation Results:")
        print(f"Mean reward: {eval_results['reward'][0]:.2f} ± {eval_results['reward'][1]:.2f}")
        print(f"Mean throughput: {eval_results['throughput'][0]:.2f} ± {eval_results['throughput'][1]:.2f}")
        print(f"Mean waiting time: {eval_results['waiting_time'][0]:.2f} ± {eval_results['waiting_time'][1]:.2f}")
        
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Make sure to close all environments
        env.close()
        if 'eval_env' in locals():
            eval_env.close()