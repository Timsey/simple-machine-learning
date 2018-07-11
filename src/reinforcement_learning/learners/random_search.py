"""
Agent that learns by random search. Works for environments with action spaces
of size 2 (binary).
"""

import numpy as np

from environments.cartpole import CartPole


NUM_POLICIES = 500
MAX_T = 100


class RandomSearch(object):
    def __init__(self, env):
        self.action_space = env.action_space
        self.state_space = env.state_space

        self.action_names = self.action_space.keys()
        self.policy = None
        self.best_policy = None
        self.best_score = 0

        self.num_policies = NUM_POLICIES
        self.max_t = MAX_T

    def get_policy(self):
        # Random policy parametrised by a vector w
        # Learns a decision boundary (state w^T) on binary action space
        weight = np.random.uniform(-1, 1, size=len(self.state_space))
        bias = np.random.uniform(-1, 1)
        return (weight, bias)

    def policy_to_action(self):
        if np.dot(self.policy[0], self.obs) + self.policy[1] > 0:
            return self.action_space[self.action_names[0]]
        else:
            return self.action_space[self.action_names[1]]

    def episode(self):
        total_reward = 0
        self.obs = env.reset()

        # Run for a maximum of max_t timesteps
        for i in range(self.max_t):
            action = policy_to_action()
            self.obs, reward, done = self.env.step(action)
            total_reward += reward
            if done:
                break

        return total_reward

    def train(self, num_policies=None, max_t=None):
        if num_policies is not None:
            self.num_policies = num_policies
        if max_t is not None:
            self.max_t = max_t

        self.best_score = 0
        # Try num_policies random policies
        for i in range(self.num_policies):
            self.policy = self.get_policy()
            score = self.episode()
            if score > self.best_score:
                self.best_score = score
                self.best_policy = self.policy


if __name__ == '__main__':
    env = CartPole()
    learner = RandomSearch(env)
    learner.train()
