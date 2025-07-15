# Reinforcement Learning for Traffic Light Optimization

This project implements a Reinforcement Learning (RL) solution to optimize traffic flow at intersections using SUMO (Simulation of Urban Mobility) and Gymnasium. The system trains a Deep Q-Network (DQN) agent to control traffic lights by minimizing waiting times, queue lengths, and maximizing throughput and average vehicle speeds.

## Key Features
- Custom OpenAI Gymnasium environment integrating SUMO traffic simulation
- Reward functions based on real-world traffic metrics:
  - Waiting time reduction
  - Queue length minimization
  - Average speed optimization
  - Throughput maximization
- DQN agent training with Stable-Baselines3
- Testing on both custom-built and real-world mapped networks (Braamfontein and Johannesburg CBD)

## Technologies Used
- **SUMO** (1.21) - Microscopic traffic simulation
- **Python** (3.12.0) - Core implementation language
- **Gymnasium** (0.29.1) - RL environment interface
- **Stable-Baselines3** (2.3.2) - DQN implementation
- **TraCI** (1.20.0) - SUMO-Python interaction
- **Matplotlib** (3.8.3) - Visualization of results

## Methodology

### 1. Environment Development
- Created custom 4-intersection network using SUMO's netedit
- Configured traffic lights with 4-phase cycles (green, yellow, red, yellow)
- Implemented one-way traffic flows with controlled entry/exit points
- Generated consistent traffic flow (500 cars/hour on all routes)

### 2. Gymnasium Integration
- Built custom environment class extending Gymnasium's interface
- Implemented action space (binary control for each intersection)
- Designed observation space (queue length, waiting time, speed, throughput)
- Developed reward function combining multiple traffic metrics

### 3. Agent Training
- Implemented DQN agent with Stable-Baselines3
- Configured network architecture (2 hidden layers, 256 units each)
- Set training parameters:
  ```python
  learning_rate = 1e-3
  buffer_size = 100000
  batch_size = 256
  exploration_fraction = 0.3
  gamma = 0.99
