"""
Agent that learns by random search. Works for environments with action spaces
of size 2 (binary).
"""

import sys
import numpy as np

sys.path.append("..")  # noqa: F401
from environments.cartpole import CartPoleEnv


NUM_POLICIES = 500
MAX_T = 100


class RandomSearch(object):
    def __init__(self, env):
        self.env = env
        self.action_space = self.env.action_space
        self.state_space = self.env.state_space

        self.policy = None
        self.best_policy = None
        self.best_score = 0
        self.do_evaluation = False
        self.do_show = False

        self.num_policies = NUM_POLICIES
        self.max_t = MAX_T

        self.max_reward = self.env.max_reward_per_episode * self.max_t

    def get_policy(self):
        # Random policy parametrised by a vector w
        # Learns a decision boundary (state w^T) on binary action space
        weight = np.random.uniform(-1, 1, size=len(self.state_space))
        bias = np.random.uniform(-1, 1)
        return (weight, bias)

    def policy_to_action(self):
        if np.dot(self.policy[0], self.obs) + self.policy[1] > 0:
            return self.action_space[0]
        else:
            return self.action_space[1]

    def episode(self):
        if self.do_evaluation:
            self.policy = self.best_policy

        total_reward = 0
        self.obs = env.reset()

        # Run for a maximum of max_t timesteps
        for i in range(self.max_t):
            action = self.policy_to_action()
            self.obs, reward, done = self.env.step(action)
            total_reward += reward

            if self.do_show:
                self.env.render_obs(self.obs)

            if done:
                break

        return total_reward

    def train(self, num_policies=None, max_t=None):
        if num_policies is not None:
            self.num_policies = num_policies
        if max_t is not None:
            self.max_t = max_t
            self.max_reward = self.env.max_reward_per_episode * self.max_t

        self.best_score = 0
        # Try num_policies random policies
        for i in range(self.num_policies):
            self.policy = self.get_policy()
            score = self.episode()
            if score > self.best_score:
                self.best_score = score
                self.best_policy = self.policy

    def evaluate(self, do_show=False):
        self.do_evaluation = True
        self.do_show = do_show
        score = self.episode()
        print('\nLearned policy got score {}/{}'.format(score,
                                                        self.max_reward))
        return score


if __name__ == '__main__':
    env = CartPoleEnv()
    learner = RandomSearch(env)

    learner.train()
    learner.evaluate(do_show=False)
