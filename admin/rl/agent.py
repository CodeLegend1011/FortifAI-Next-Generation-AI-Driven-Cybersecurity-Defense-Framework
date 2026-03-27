"""
admin/rl/agent.py

Real Reinforcement Learning Threat Response Loop (Tabular Q-Learning).
Learns the optimal responses to different threat states by balancing 
analyst feedback (rewards for correct containment, penalties for false positives).
"""

from typing import Dict, List, Any
import time
import numpy as np
import random
import os
import json

try:
    from shared.detection_config import (
        RL_ACTIONS, RL_LEARNING_RATE, RL_GAMMA, RL_EPSILON,
        RL_FEEDBACK_REWARD_MAGNITUDE,
        RL_SCORE_HIGH_THRESHOLD, RL_SCORE_MED_THRESHOLD,
        RL_Q_INIT_RANGE,
    )
except ImportError:
    RL_ACTIONS = ["ignore", "block_ip", "kill_process", "isolate_network"]
    RL_LEARNING_RATE = 0.1
    RL_GAMMA = 0.9
    RL_EPSILON = 0.1
    RL_FEEDBACK_REWARD_MAGNITUDE = 5.0
    RL_SCORE_HIGH_THRESHOLD = 8.0
    RL_SCORE_MED_THRESHOLD = 5.0
    RL_Q_INIT_RANGE = 0.01

class RLAgent:
    def __init__(self, state_dim: int = 24, action_dim: int = 4, q_table_path: str = "q_table.json"):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.is_trained = False
        
        self.alpha = RL_LEARNING_RATE
        self.gamma = RL_GAMMA
        self.epsilon = RL_EPSILON
        
        self.actions = list(RL_ACTIONS)
        self.q_table = {}
        
        self.last_state = None
        self.last_action = None
        
        self.q_table_path = q_table_path
        self._load_q_table()
        
    def _extract_state(self, threat_state: Dict[str, Any]) -> str:
        """Discretize continuous/complex alerts into a manageable state space."""
        # Simple state representations based on severity and category
        severity = str(threat_state.get('severity', 'unknown')).lower()
        category = str(threat_state.get('category', 'unknown')).lower()
        
        # Threat level bounds
        score = threat_state.get('ensemble_score', 0)
        if score > RL_SCORE_HIGH_THRESHOLD:
            level = 'high_confidence'
        elif score > RL_SCORE_MED_THRESHOLD:
            level = 'medium_confidence'
        else:
            level = 'low_confidence'
            
        return f"{severity}_{category}_{level}"
        
    def _get_q_values(self, state_key: str) -> np.ndarray:
        if state_key not in self.q_table:
            # Initialize with small random values to break ties
            self.q_table[state_key] = np.random.uniform(low=-RL_Q_INIT_RANGE, high=RL_Q_INIT_RANGE, size=len(self.actions)).tolist()
        return np.array(self.q_table[state_key])

    def get_action(self, threat_state: Dict[str, Any], explore: bool = True) -> str:
        """
        Epsilon-greedy action selection based on the Q-table.
        """
        state_key = self._extract_state(threat_state)
        self.last_state = state_key
        
        # Epsilon-greedy exploration
        if explore and random.random() < self.epsilon:
            action_idx = random.randint(0, len(self.actions) - 1)
        else:
            q_values = self._get_q_values(state_key)
            action_idx = int(np.argmax(q_values))
            
        self.last_action = self.actions[action_idx]
        
        print(f"[RL] Evaluating state: {state_key} | Action chosen: {self.last_action}")
        return self.last_action
        
    def step(self, action: str, reward: float, next_state: Dict[str, Any], done: bool):
        """
        Update the Q-table (Bellman Equation) based on feedback.
        Requires that get_action was called previously to set self.last_state.
        """
        if not self.last_state:
            print("[RL] Warning: step() called without previous get_action().")
            return
            
        try:
            action_idx = self.actions.index(action)
        except ValueError:
            print(f"[RL] Unknown action {action}")
            return
            
        current_q = self._get_q_values(self.last_state)
        
        if next_state and not done:
            next_state_key = self._extract_state(next_state)
            next_q = self._get_q_values(next_state_key)
            max_next_q = np.max(next_q)
        else:
            max_next_q = 0.0
            
        # Q-Learning update rule
        current_q[action_idx] = current_q[action_idx] + self.alpha * (reward + self.gamma * max_next_q - current_q[action_idx])
        
        # Save back to table
        self.q_table[self.last_state] = current_q.tolist()
        
        if reward > 0:
            print(f"[RL] Positive reinforcement: Action '{action}' on '{self.last_state}' (+{reward})")
        else:
            print(f"[RL] Negative reinforcement: Action '{action}' on '{self.last_state}' ({reward})")
            
        self._save_q_table()
        
        if done:
            self.last_state = None
            self.last_action = None

    def train_offline(self, historical_incidents: List[Dict[str, Any]]):
        """
        Iterate over historical SOC tickets and pre-populate the Q-Table.
        """
        print(f"[RL] Pre-training on {len(historical_incidents)} historical incidents...")
        
        for incident in historical_incidents:
            state = incident.get("state", {})
            action = incident.get("action", "ignore")
            reward = incident.get("reward", 0.0)
            
            # Simple 1-step offline learning
            state_key = self._extract_state(state)
            self.last_state = state_key
            self.step(action, reward, None, True)
            
        self.is_trained = True
        print(f"[RL] Training complete. Q-Table Size: {len(self.q_table)}")

    def _save_q_table(self):
        try:
            with open(self.q_table_path, 'w') as f:
                json.dump(self.q_table, f)
        except Exception as e:
            print(f"[RL] Could not save Q-table: {e}")
            
    def _load_q_table(self):
        if os.path.exists(self.q_table_path):
            try:
                with open(self.q_table_path, 'r') as f:
                    self.q_table = json.load(f)
                self.is_trained = True
                print(f"[RL] Loaded Q-Table with {len(self.q_table)} states.")
            except Exception as e:
                print(f"[RL] Error loading Q-table: {e}")

    def provide_feedback(self, threat_state: Dict[str, Any], action: str, was_correct: bool):
        """
        Human-in-the-loop feedback: analyst rates whether the recommended
        action was appropriate.  +reward for correct, -penalty for wrong.
        """
        reward = RL_FEEDBACK_REWARD_MAGNITUDE if was_correct else -RL_FEEDBACK_REWARD_MAGNITUDE
        self.last_state = self._extract_state(threat_state)
        self.step(action, reward, None, done=True)
        print(f"[RL] Analyst feedback: action='{action}' correct={was_correct} reward={reward}")


if __name__ == "__main__":
    agent = RLAgent(q_table_path="test_q_table.json")
    
    # Offline dummy data
    history = [
        {"state": {"severity": "critical", "category": "network", "ensemble_score": 9.5}, "action": "isolate_network", "reward": 10.0},
        {"state": {"severity": "high", "category": "process", "ensemble_score": 7.5}, "action": "kill_process", "reward": 5.0},
        {"state": {"severity": "high", "category": "file", "ensemble_score": 6.0}, "action": "isolate_network", "reward": -5.0}, # FP
    ]
    
    agent.train_offline(history)
    
    # Live inference
    test_state = {"severity": "critical", "category": "network", "ensemble_score": 9.8}
    
    # Force exploitation (no random exploration) for test
    agent.epsilon = 0.0 
    
    action = agent.get_action(test_state)
    print(f"\\nFinal Recommended action for critical network event: {action}")
