"""
Base learner on which other reinforcement learners are based.
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
        self.env_info = env.env_info

        self.best_policy = None
        self.best_score = 0

        # Time to run the environment for during training and evaluation
        self.max_t = MAX_T
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

    def policy_to_action(self, policy, obs):
        """
        Gets an action from a policy. Currently works for policies that
        define binary hyperplanes (binary, e.g. binary CartPole), and policies
        defined on a grid (grid, e.g. FrozenLake).

        Arguments:
        - policy: np.array, represents the policy to run.
        - obs: np.array, current observation of the environment.

        Returns:
        - action: int, represents action to take.
        """

        if self.env_info['policy_type'] == 'binary':
            # A policy defines an action for every state vector
            # An action is chosen according to a simple hyperplane
            if np.dot(policy[:obs.size], obs) + policy[-1] > 0:
                ind = 0
            else:
                ind = 1
            return self.env_info['action_space'][ind]

        elif self.env_info['policy_type'] == 'grid':
            # A policy defines an action for every point in the grid
            # obs (observation) is a position index in the grid
            return policy[obs]

        elif self.env_info['policy_type'] == 'function':
            # A policy is a function that maps states to actions
            # Every state gets assigned an action (e.g. through softmax)
            raise NotImplementedError('This state type is not yet supported.')

    def episode(self, policy, do_show=False, max_t=MAX_T):
        """
        Runs an episode of the policy in the environment.

        Arguments:
        - policy: np.array, represents the policy to run.
        - do_show: bool, whether to visualise the episode during evaluation.
        - max_t: number of steps in an episode.

        Returns:
        - total_reward: float, reward earned during the episode.
        """
        self.max_t = max_t

        total_reward = 0
        obs = self.env.reset()

        # Run for a maximum of max_t timesteps
        for i in range(self.max_t):
            action = self.policy_to_action(policy, obs)
            obs, reward, done = self.env.step(action)
            total_reward += reward

            if do_show:
                self.env.render_obs(obs)

            if done:  # Episode ended
                self.env.done = True
                break

        # Episode ended
        self.env.done = True

        return total_reward

    def evaluate(self, policy, do_print=True, do_show=False, average=0,
                 max_t=MAX_T):
        """
        Evaluates the given policy.

        Arguments:
        - policy: np.array, represents the policy to evaluate.
        - do_print: bool, whether to print evaluation results.
        - do_show: bool, whether to visualise the episode during evaluation.
        - average: int, how many episodes to average to get a score. Values
                   greater than 1 for this argument cannot be combined with
                   do_show.
        - max_t: int, number of timesteps in an episode.

        Returns:
        - score: float, (average) score of the policy.
        """

        assert isinstance(average, int) and average >= 0, 'average should be '
        'a positive integer or zero!'

        if policy is None:
            raise ValueError("Policy is None; it may be the case that the "
                             "learner found no policy with score above 0.")
        self.max_t = max_t

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
        pass
