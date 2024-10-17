import gymnasium as gym
from gymnasium import spaces
import numpy as np
import traci
import os
import sys
from typing import Dict, List, Tuple
from collections import deque

class SumoIntersectionEnv(gym.Env):
    def __init__(self, gui: bool = False):
        super(SumoIntersectionEnv, self).__init__()
        
        if 'SUMO_HOME' in os.environ:
            tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
            sys.path.append(tools)
        else:
            sys.exit("Please declare environment variable 'SUMO_HOME'")
        
        self.sim_max_time = 3600  # 1 hour in seconds
        self.delta_time = 10  # Action step time in seconds
        self.max_speed = 13.89  # 50 km/h in m/s
        
        # Simulation configuration
        self.sumo_config = "intersection.sumocfg"
        self.junction_ids = ["junct1", "junct2", "junct3", "junct4"]
        
        # Discrete action space
        self.action_space = spaces.Discrete(16)  # 2^4 possible combinations
        
        # Observation space setup
        num_edges = 12
        self.obs_per_edge = 4  # queue length, waiting time, average speed, density
        obs_size = num_edges * self.obs_per_edge
        self.observation_space = spaces.Box(
            low=np.zeros(obs_size),
            high=np.ones(obs_size),
            dtype=np.float32
        )
        
        self.edge_ids = [
            "North", "South", "East", "West",
            "North2", "south2", "Lower_East", "lower_west",
            "end_east1", "end_east2", "end_south1", "end_south2"
        ]
        
        # Define exit edges specifically
        self.exit_edges = ["end_east1", "end_east2", "end_south1", "end_south2"]
        
        self.sumo_binary = "sumo-gui" if gui else "sumo"
        self.label = "Training" if not gui else "Visualization"
        
        # Enhanced metrics tracking
        self.metrics_window = 100
        self.episode_metrics = {
            'waiting_times': deque(maxlen=self.metrics_window),
            'queue_lengths': deque(maxlen=self.metrics_window),
            'throughput': deque(maxlen=self.metrics_window),
            'average_speeds': deque(maxlen=self.metrics_window),
            'rewards': deque(maxlen=self.metrics_window)
        }
        
        # Performance tracking
        self.vehicles_exited = set()
        self.cumulative_reward = 0
        self.steps = 0
        self.previous_vehicle_ids = set()
        
    def start_simulation(self):
        sumo_cmd = [
            self.sumo_binary,
            "-c", self.sumo_config,
            "--no-step-log", "true",
            "--no-warnings", "true",
            "--random", "true",
            "--start", "true",
            "--quit-on-end", "true",
            "--waiting-time-memory", "1000",
            "--log", f"{self.label}.log"
        ]
        traci.start(sumo_cmd)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self.steps += 1
        
        # Convert single integer action to binary actions for each junction
        junction_actions = self._decode_action(action)
        
        # Apply actions and collect pre-action metrics
        pre_metrics = self._get_metrics()
        self._apply_actions(junction_actions)
        
        # Simulate for delta_time
        for _ in range(self.delta_time):
            traci.simulationStep()
        
        # Collect post-action metrics and compute reward
        post_metrics = self._get_metrics()
        reward = self._compute_reward(pre_metrics, post_metrics)
        
        # Update episode metrics
        self._update_metrics(post_metrics, reward)
        
        # Get new state
        observation = self._get_state()
        
        # Check termination
        current_time = traci.simulation.getTime()
        terminated = current_time >= self.sim_max_time
        truncated = False
        
        info = self._get_info(post_metrics)
        return observation, reward, terminated, truncated, info

    def _compute_reward(self, pre_metrics: Dict, post_metrics: Dict) -> float:
        # Normalized reward components
        waiting_time_change = (pre_metrics['waiting_time'] - post_metrics['waiting_time']) / 60.0
        queue_change = (pre_metrics['queue_length'] - post_metrics['queue_length']) / 10.0
        speed_reward = post_metrics['average_speed'] / self.max_speed
        throughput_reward = post_metrics['new_exits'] / 5.0  # Normalize by expected max exits per step
        
        # Combine rewards with weights
        reward = (
            0.35 * waiting_time_change +
            0.25 * queue_change +
            0.15 * speed_reward +
            0.25 * throughput_reward
        )
        
        # Clip reward for stability
        return float(np.clip(reward, -1.0, 1.0))

    def _get_state(self) -> np.ndarray:
        state = []
        for edge in self.edge_ids:
            try:
                queue_length = traci.edge.getLastStepHaltingNumber(edge)
                waiting_time = traci.edge.getWaitingTime(edge)
                mean_speed = traci.edge.getLastStepMeanSpeed(edge)
                vehicle_count = traci.edge.getLastStepVehicleNumber(edge)
                edge_length = traci.lane.getLength(edge + "_0")  # Assuming at least one lane
                density = vehicle_count / edge_length if edge_length > 0 else 0
                
                # Normalize values
                normalized_queue = min(queue_length / 10.0, 1.0)
                normalized_wait = min(waiting_time / 60.0, 1.0)
                normalized_speed = mean_speed / self.max_speed
                normalized_density = min(density / 0.1, 1.0)  # Assuming max density of 0.1 veh/m
                
                state.extend([normalized_queue, normalized_wait, normalized_speed, normalized_density])
            except traci.exceptions.TraCIException:
                state.extend([0, 0, 0, 0])
        
        return np.array(state, dtype=np.float32)

    def _get_metrics(self) -> Dict:
        try:
            total_waiting_time = sum(traci.edge.getWaitingTime(edge) for edge in self.edge_ids)
            total_queue = sum(traci.edge.getLastStepHaltingNumber(edge) for edge in self.edge_ids)
            
            # Calculate average speed across all edges
            speeds = [traci.edge.getLastStepMeanSpeed(edge) for edge in self.edge_ids]
            valid_speeds = [s for s in speeds if s >= 0]
            avg_speed = np.mean(valid_speeds) if valid_speeds else 0
            
            # Calculate throughput based on vehicles that have exited
            current_exited = set()
            for edge in self.exit_edges:
                exited_vehicles = set(traci.edge.getLastStepVehicleIDs(edge))
                current_exited.update(exited_vehicles)
            
            new_exits = len(current_exited - self.vehicles_exited)
            self.vehicles_exited = current_exited
            
            return {
                'waiting_time': total_waiting_time,
                'queue_length': total_queue,
                'average_speed': avg_speed,
                'new_exits': new_exits,
                'total_exits': len(self.vehicles_exited)
            }
        except traci.exceptions.TraCIException:
            return {
                'waiting_time': 0,
                'queue_length': 0,
                'average_speed': 0,
                'new_exits': 0,
                'total_exits': len(self.vehicles_exited)
            }

    def _update_metrics(self, metrics: Dict, reward: float):
        self.episode_metrics['waiting_times'].append(metrics['waiting_time'])
        self.episode_metrics['queue_lengths'].append(metrics['queue_length'])
        self.episode_metrics['throughput'].append(metrics['new_exits'])
        self.episode_metrics['average_speeds'].append(metrics['average_speed'])
        self.episode_metrics['rewards'].append(reward)
        self.cumulative_reward += reward

    def _get_info(self, metrics: Dict) -> Dict:
        return {
            'current_metrics': metrics,
            'moving_averages': {
                'waiting_time': np.mean(self.episode_metrics['waiting_times']) if self.episode_metrics['waiting_times'] else 0,
                'queue_length': np.mean(self.episode_metrics['queue_lengths']) if self.episode_metrics['queue_lengths'] else 0,
                'throughput': np.mean(self.episode_metrics['throughput']) if self.episode_metrics['throughput'] else 0,
                'average_speed': np.mean(self.episode_metrics['average_speeds']) if self.episode_metrics['average_speeds'] else 0,
                'reward': np.mean(self.episode_metrics['rewards']) if self.episode_metrics['rewards'] else 0
            },
            'cumulative_reward': self.cumulative_reward,
            'step': self.steps
        }

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        
        if traci.isLoaded():
            traci.close()
        
        self.start_simulation()
        
        # Reset metrics and counters
        self.vehicles_exited = set()
        self.cumulative_reward = 0
        self.steps = 0
        
        for key in self.episode_metrics:
            self.episode_metrics[key].clear()
        
        return self._get_state(), self._get_info(self._get_metrics())

    def _decode_action(self, action: int) -> List[int]:
        return [(action >> i) & 1 for i in range(len(self.junction_ids))]

    def _apply_actions(self, actions: List[int]):
        for i, junction_id in enumerate(self.junction_ids):
            if actions[i] == 1:
                try:
                    current_phase = traci.trafficlight.getPhase(junction_id)
                    next_phase = (current_phase + 1) % len(traci.trafficlight.getAllProgramLogics(junction_id)[0].phases)
                    traci.trafficlight.setPhase(junction_id, next_phase)
                except traci.exceptions.TraCIException:
                    continue

    def close(self):
        if traci.isLoaded():
            traci.close()