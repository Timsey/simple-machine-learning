"""
Agent that learns by random search. Works for environments with action spaces
of size 2 (binary).
"""

import sys
import numpy as np
from base_learner import BaseLearner

sys.path.append("..")  # noqa: F401
from environments.cartpole import CartPoleEnv

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

        if self.state_space['state_type'] == 'continuous':
            # Random policy parametrised by a vector w
            # Learns a decision boundary (state w^T) on binary action space
            policy = np.random.uniform(-1, 1, size=self.state_space[
                'feature_size'] + 1)
        elif self.state_space['state_type'] == 'discrete':
            # Random policy that gives an action (int) for every possible state
            policy = np.random.choice(self.action_space,
                                      size=self.state_space['feature_size'])
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
            self.max_reward = self.env.max_reward_per_episode * self.max_t

        # Try num_policies random policies
        for i in range(self.num_policies):
            policy = self.get_policy()
            score = self.episode(policy)
            # Keep track of best policy
            if score > self.best_score:
                self.best_score = score
                self.best_policy = policy


if __name__ == '__main__':
    env = CartPoleEnv()
    learner = RandomSearch(env)

    learner.train(num_policies=500, max_t=100)
    learner.evaluate(learner.best_policy, do_show=False, average=100)
