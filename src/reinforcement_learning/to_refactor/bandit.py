import numpy as np

np.random.seed(42)

num_actions = 10
burn_in = 10000
timesteps = 1000
eps = 0.1

actions = list(range(num_actions))
action_values = np.random.normal(size=num_actions)
actions_taken = np.zeros(num_actions)
estimated_values = np.zeros(num_actions)

total_reward = 0
for t in range(timesteps + burn_in):
    action = np.argmax(action_values)
    # Burn-in epsilon greedy (exploration + exploitation)
    # Run agent greedy after burn-in (pure exploitation)
    if t < burn_in:
        if np.random.uniform() < eps:
            action = np.random.randint(0, num_actions)

    reward = action_values[action] + np.random.normal()
    total_reward += reward
    estimated_values[action] = (actions_taken[action] * estimated_values[action] + reward) / (actions_taken[action] + 1)
    actions_taken[action] += 1

print("\nValue estimation after {} timesteps:\n".format(timesteps))
print("True   Estimated   Absolute difference")
for i, j in zip(action_values, estimated_values):
    print("{:>5.2f} {:>8.2f} {:>8.2f}".format(i, j, np.abs(i - j)))

print("\nTotal error: {:.3f}".format(np.sum(np.abs(action_values - estimated_values))))

print("\nTotal reward: {:.3f}".format(total_reward))
