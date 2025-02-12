import matplotlib.pyplot as plt
from tqdm import tqdm  # Ensure progress bar works
from bandit import run_experiment

print("Starting ε-greedy experiment with multiple epsilon values...")

# Different epsilon values to test
epsilon_values = [0.01, 0.1, 0.5]

for epsilon in epsilon_values:
    print(f"Running experiment for ε = {epsilon}...")

    # Run experiments for both methods
    rewards_sample, optimal_sample = run_experiment(method='sample-average', epsilon=epsilon)
    rewards_alpha, optimal_alpha = run_experiment(method='constant-alpha', alpha=0.1, epsilon=epsilon)

    print(f"Experiment for ε = {epsilon} complete. Generating plots...")

    # Plot results for average reward
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_sample, label=f'Sample-Average (ε={epsilon})')
    plt.plot(rewards_alpha, label=f'Constant Step-Size (α=0.1, ε={epsilon})')
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.title(f'Average Reward Comparison for ε={epsilon}')
    plt.savefig(f"average_reward_epsilon_{epsilon}.png")  # Save the plot
    plt.show()

    # Plot results for optimal action selection
    plt.figure(figsize=(10, 5))
    plt.plot(optimal_sample, label=f'Sample-Average (ε={epsilon})')
    plt.plot(optimal_alpha, label=f'Constant Step-Size (α=0.1, ε={epsilon})')
    plt.xlabel('Steps')
    plt.ylabel('% Optimal Action')
    plt.legend()
    plt.title(f'Optimal Action % Over Time for ε={epsilon}')
    plt.savefig(f"optimal_action_epsilon_{epsilon}.png")  # Save the plot
    plt.show()

print("All experiments complete.")
