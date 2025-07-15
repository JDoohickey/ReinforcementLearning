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
## Technology Stack

### Core Technologies
<p align="left">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/SUMO-E68310?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA1MTIgNTEyIj48cGF0aCBmaWxsPSIjZmZmIiBkPSJNMjU2IDhDMTE5IDggOCAxMTkgOCAyNTZzMTExIDI0OCAyNDggMjQ4IDI0OC0xMTEgMjQ4LTI0OFMzOTMgOCAyNTYgOHptMCA0NDhjLTExMC41IDAtMjAwLTg5LjUtMjAwLTIwMFMxNDUuNSA1NiAyNTYgNTZzMjAwIDg5LjUgMjAwIDIwMC04OS41IDIwMC0yMDAgMjAwem0xMDEuOC0yNjEuN0wtMzE3LjggNDQxYy0zLjEgMy4xLTguMiAzLjEtMTEuMyAwbC0zNC0zNGMtMy4xLTMuMS0zLjEtOC4yIDAtMTEuM2wxMjcuMy0xMjcuM0w0MyAxMTkuN2MtMy4xLTMuMS0zLjEtOC4yIDAtMTEuM2wzNC0zNGMzLjEtMy4xIDguMi0zLjEgMTEuMyAwbDEyNy4zIDEyNy4zTDM5My43IDQzYzMuMS0zLjEgOC4yLTMuMSAxMS4zIDBsMzQgMzRjMy4xIDMuMSAzLjEgOC4yIDAgMTEuM2wtMTI3LjMgMTI3LjNMNTAxIDE5OS43YzMuMSAzLjEgMy4xIDguMiAwIDExLjNsLTM0IDM0Yy0zLjEgMy4xLTguMiAzLjEtMTEuMyAweiIvPjwvc3ZnPg==" alt="SUMO">
  <img src="https://img.shields.io/badge/Gymnasium-6F6F6F?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0iI2ZmZiIgZD0iTTEyIDBDNS4zNzMgMCAwIDUuMzczIDAgMTJzNS4zNzMgMTIgMTIgMTIgMTItNS4zNzMgMTItMTJTMTguNjI3IDAgMTIgMHptMCAyMmMtNS41MjMgMC0xMC00LjQ3Ny0xMC0xMHM0LjQ3Ny0xMCAxMC0xMCAxMCA0LjQ3NyAxMCAxMC00LjQ3NyAxMC0xMCAxMHptLTUtMTBoMnY3aC0ydi03em0zIDBoMnY3aC0ydi03em0zIDBoMnY3aC0ydi03eiIvPjwvc3ZnPg==" alt="Gymnasium">
  <img src="https://img.shields.io/badge/Stable_Baselines3-000000?style=for-the-badge&logo=stable-baselines3&logoColor=white" alt="Stable Baselines3">
</p>

### Development Tools
<p align="left">
  <img src="https://img.shields.io/badge/Visual_Studio_Code-007ACC?style=for-the-badge&logo=visual-studio-code&logoColor=white" alt="VSCode">
  <img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub">
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=Jupyter&logoColor=white" alt="Jupyter">
</p>

### Key Libraries
<p align="left">
  <img src="https://img.shields.io/badge/TraCI-FFD43B?style=for-the-badge&logo=python&logoColor=blue" alt="TraCI">
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white" alt="Matplotlib">
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas">
</p>
