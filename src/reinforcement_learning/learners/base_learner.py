"""
Agent that learns by random search. Works for environments with action spaces
of size 2 (binary).
"""

import sys
import numpy as np


# Number of steps in an episode
MAX_T = 100


class BaseLearner(object):
    def __init__(self, env):
        """
        Base class for reinforcement learners.

        Arguments:
        - env: instance of an environment to run the learner in.
        """

        self.env = env
        self.action_space = self.env.action_space
        self.state_space = self.env.state_space

        self.best_policy = None
        self.best_score = 0

        # Time to run the environment for during training and evaluation
        self.max_t = MAX_T
        # Maximum possible reward given the environment
        self.max_reward = self.env.max_reward_per_episode * self.max_t

    def policy_to_action(self, policy, obs):
        """
        Gets an action from a policy. Currently works for policies that
        define hyperplanes (continuous, e.g. binary CartPole), and policies
        defined on a grid (discrete, e.g. FrozenLake).

        Arguments:
        - policy: np.array, represents the policy to run.
        - obs: np.array, current observation of the environment.

        Returns:
        - action: int, represents action to take.
        """

        if self.state_space['state_type'] == 'continuous':
            # In the continuous case, we separate actions using a hyperplane
            if np.dot(policy[:obs.size], obs) + policy[-1] > 0:
                return self.action_space[0]
            else:
                return self.action_space[1]
        elif self.state_space['state_type'] == 'discrete':
            # In the discrete case, we separate actions by grid
            # obs is a position index in the grid
            return policy[obs]

    def episode(self, policy, do_show=False):
        """
        Runs an episode of the policy in the environment.

        Arguments:
        - policy: np.array, represents the policy to run.
        - do_show: bool, whether to visualise the episode during evaluation.

        Returns:
        - total_reward: float, reward earned during the episode.
        """

        total_reward = 0
        obs = self.env.reset()

        # Run for a maximum of max_t timesteps
        for i in range(self.max_t):
            action = self.policy_to_action(policy, obs)
            obs, reward, done = self.env.step(action)
            total_reward += reward

            if do_show:
                self.env.render_obs(obs)

            if done:
                break

        return total_reward

    def evaluate(self, policy, do_print=True, do_show=False, average=0):
        """
        Evaluates the given policy.

        Arguments:
        - policy: np.array, represents the policy to evaluate.
        - do_print: bool, whether to print evaluation results.
        - do_show: bool, whether to visualise the episode during evaluation.
        - average: int, how many episodes to average to get a score. Values
                   greater than 1 for this argument cannot be combined with
                   do_show.

        Returns:
        - score: float, (average) score of the policy.
        """

        assert isinstance(average, int) and average >= 0, 'average should be '
        'a positive integer or zero!'
        if average not in [0, 1]:
            if do_show:
                print('WARNING: do_show has no effect while average is > 1. '
                      'Set average in [0, 1] to see animation.')
            total_score = 0
            for _ in range(average):
                total_score += self.episode(policy)
            score = total_score / average
        else:
            score = self.episode(policy, do_show=do_show)

        if do_print:
            print('\nLearned policy got score {}/{}'.format(score,
                                                            self.max_reward))
        return score

    def get_policy(self):
        """
        Should return a policy by the strategy of the learner.
        """

        return

    def train(self, max_t=None):
        """
        Should train the learner, i.e. find a good policy.

        Arguments:
        - max_t: int, number of steps in an episode.
        """

        # Always include this in train() functions: it allows evaluate to find
        # the proper maximum score.
        if max_t is not None:
            self.max_t = max_t
            self.max_reward = self.env.max_reward_per_episode * self.max_t
        pass
