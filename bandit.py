import numpy as np
from tqdm import tqdm  # Ensure progress bar displays correctly

class NonstationaryBandit:
    def __init__(self, k=10, std_dev=0.01):
        """ Initializes a nonstationary k-armed bandit problem """
        self.k = k  # Number of arms
        self.q_star = np.zeros(k)  # True action values, start at 0
        self.std_dev = std_dev  # Standard deviation for random walk
    
    def step(self, action):
        """ Returns a reward for the selected action and updates true action values """
        reward = np.random.normal(self.q_star[action], 1)  # Reward from N(q*(a), 1)
        self.q_star += np.random.normal(0, self.std_dev, self.k)  # Random walk for all arms
        return reward

def run_experiment(method='sample-average', alpha=0.1, epsilon=0.1, k=10, steps=10000, runs=2000):
    """ Runs the reinforcement learning experiment for both methods with a loading bar """
    avg_rewards = np.zeros(steps)
    optimal_action_counts = np.zeros(steps)
    
    for run in tqdm(range(runs), desc="Running simulations", unit="run"):
        bandit = NonstationaryBandit(k)
        Q = np.zeros(k)  # Estimated action values
        N = np.zeros(k)  # Action counts
        
        for t in range(steps):
            optimal_action = np.argmax(bandit.q_star)  # Track best action
            
            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = np.random.randint(k)  # Explore
            else:
                action = np.argmax(Q)  # Exploit
            
            reward = bandit.step(action)  # Get reward from environment
            
            # Update estimates
            if method == 'sample-average':
                N[action] += 1
                Q[action] += (reward - Q[action]) / N[action]  # Sample-average update
            else:
                Q[action] += alpha * (reward - Q[action])  # Constant step-size update
            
            avg_rewards[t] += reward
            optimal_action_counts[t] += (action == optimal_action)
    
    # Average over runs
    avg_rewards /= runs
    optimal_action_counts = (optimal_action_counts / runs) * 100  # Convert to percentage
    
    return avg_rewards, optimal_action_counts
