"""
Agent that learns by random search.

Based on blog post by Moustafa Alzantot:
https://medium.com/@m.alzantot/deep-reinforcement-learning-demystified-episode-0-2198c05a6124
"""

import sys
import numpy as np
from base_learner import BaseLearner

sys.path.append("..")  # noqa: F401
from environments.cartpole import CartPoleEnv
from environments.frozenlake import FrozenLakeEnv

# Number of random policies to generate during training
NUM_POLICIES = 500


class RandomSearch(BaseLearner):
    def __init__(self, env):
        super().__init__(env)

        # Number of random policies to generate
        self.num_policies = NUM_POLICIES

    def get_policy(self):
        """
        Generates a random policy. Currently works for policies that
        define hyperplanes (continuous, e.g. binary CartPole), and policies
        defined on a grid (discrete, e.g. FrozenLake).

        Returns:
        - policy: np.array, represents the generated policy.
        """

        if self.env_info['policy_type'] == 'binary':
            # Random policy parametrised by a vector w
            # Learns a decision boundary (state w^T) on binary action space
            policy = np.random.uniform(-1, 1, size=self.env_info[
                'state_size'] + 1)
        elif self.env_info['policy_type'] == 'grid':
            # Random policy that gives an action (int) for every possible state
            policy = np.random.choice(self.env_info['action_space'],
                                      size=self.env_info['state_size'])
        return policy

    def train(self, num_policies=None, max_t=None):
        """
        Train the RandomSearch learner. Sets self.best_policy to the best
        found policy.

        Training consists of randomly generating some policies, then
        evaluating them all and picking the one that scored best.

        Arguments:
        - num_policies: int, number of random policies to generate.
        - max_t: int, number of steps in an episode.
        """

        if num_policies is not None:
            self.num_policies = num_policies
        if max_t is not None:
            self.max_t = max_t
            # Maximum possible reward given the environment
            if (self.env.max_reward_per_timestep is None
                and self.env.max_reward_per_episode is None):
                raise ValueError("Either max_reward_per_timestep or "
                                 "max_reward_per_episode needs to be set.")
            elif (self.env.max_reward_per_timestep is None
                  and self.env.max_reward_per_episode is None):
                raise ValueError("Either max_reward_per_timestep or "
                                 "max_reward_per_episode needs to be None.")
            elif self.env.max_reward_per_timestep is not None:
                self.max_reward = self.env.max_reward_per_timestep * self.max_t
            else:
                self.max_reward = self.env.max_reward_per_episode

        # Try num_policies random policies
        for _ in range(self.num_policies):
            policy = self.get_policy()
            score = self.episode(policy)
            # Keep track of best policy
            if score > self.best_score:
                self.best_score = score
                self.best_policy = policy


if __name__ == '__main__':
    env = CartPoleEnv()
    # env = FrozenLakeEnv()
    learner = RandomSearch(env)

    learner.train(num_policies=500, max_t=100)
    learner.evaluate(learner.best_policy, do_show=False, average=100, max_t=20)
