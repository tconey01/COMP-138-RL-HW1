import matplotlib.pyplot as plt
from tqdm import tqdm  # Ensure progress bar is visible
from bandit import run_experiment

print("Starting standard reinforcement learning experiment...")

# Run experiments for both methods
rewards_sample, optimal_sample = run_experiment(method='sample-average')
rewards_alpha, optimal_alpha = run_experiment(method='constant-alpha', alpha=0.1)

print("Experiment complete. Generating and saving plots...")

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(rewards_sample, label='Sample-Average')
plt.plot(rewards_alpha, label='Constant Step-Size (α=0.1)')
plt.xlabel('Steps')
plt.ylabel('Average Reward')
plt.legend()
plt.title('Average Reward Comparison')
plt.savefig("average_reward_comparison.png")  # Save the plot
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(optimal_sample, label='Sample-Average')
plt.plot(optimal_alpha, label='Constant Step-Size (α=0.1)')
plt.xlabel('Steps')
plt.ylabel('% Optimal Action')
plt.legend()
plt.title('Optimal Action % Over Time')
plt.savefig("optimal_action_comparison.png")  # Save the plot
plt.show()
