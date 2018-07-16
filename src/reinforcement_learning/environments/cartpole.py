"""
CartPole environment for reinforcement learning.
"""

import math
import numpy as np
import matplotlib.pyplot as plt


class CartPoleEnv(object):
    def __init__(self):
        """
        Initialise the CartPole environment.
        """

        self.gravity = 9.81
        self.mass_cart = 1.0
        self.mass_pole = 0.1
        self.mass_total = self.mass_cart + self.mass_pole
        self.length = 0.5  # half pole length
        self.mass_length_pole = self.mass_pole * self.length
        self.force = 10  # strength of movement TODO: make learnable?
        self.tau = 0.01  # timestep of simulation

        # Failure conditions
        self.max_theta = 12 * 2 * math.pi / 360
        self.max_x = 2.4

        # Initialise environment for learning
        self.state = self.reset()

        # Define the info the learners need to interact with the environment
        self.env_info = {'policy_type': 'binary',  # separating hyperplane
                         'action_space':  [0, 1],  # left and right
                         'state_size': 4}  # x, x_dot, theta, theta_dot
        # # Example for Frozen Lake
        # self.env_info = {'policy_type': 'grid',
        #                  'action_space': [1, 2, 3, 4],
        #                  'state_size': 16}
        # # Example for (e.g.) Pong through image instead of specific features
        # self.env_info = {'policy_type': 'function',
        #                  'action_space': [0, 1],
        #                  'state_size': IMAGE_DIMENSION}

        self.max_reward_per_episode = 1.0

    def set_seed(self, seed=None):
        """
        Sets the random seed by calling np.random.seed(seed) and sets
        self.seed to the value of seed.
        """

        if seed is None:
            seed = 0
        np.random.seed(seed)
        self.seed = seed

    def get_seed(self):
        """
        Get the random seed. Note that this will fail to produce the correct
        seed if the seed was not set through this class' get_seed() function.

        Returns:
        - self.seed: last set random seed.
        """

        return self.seed

    def step(self, action):
        """
        Do a step in the simulation. Requires an action to be taken in [0, 1].

        The state is fully defined by the position and speed in the x
        direction, combined with that in the theta direction.

        Directions defined as:
        - Positive x-direction: right
        - Positive theta-direction: clockwise
        - Positive F-direction: right
        - Action 0 is left, 1 is right

        Arguments:
        - action: one of 0 or 1, represents the action to take (left or right).

        Returns:
        - self.state: np.array, represents the current state of the
                      environment.
        - reward: float, the reward earned during this step.
        - done: bool, whether the episode should finish after this step (i.e.
                the pole has dropped too far).
        """

        assert action in (0, 1), "action must be 0 (left) or 1 (right)"

        # Check if already finished
        if self.done:
            print('WARNING: calling step() even though the environment has '
                  'already returned done=True. Call reset() first before '
                  'step() again: anything beyond this is undefined behaviour.')

        # Get current state
        state = self.state
        x, x_dot, theta, theta_dot = state

        # Update position using velocity in previous step
        x = x + x_dot * self.tau
        theta = theta + theta_dot * self.tau

        # Defines force direction
        force = self.force if action == 1 else -self.force

        cos_theta, sin_theta = math.cos(theta), math.sin(theta)

        # This term appears in the expression for both x_acc and theta_acc
        common = (force + self.mass_length_pole * sin_theta *
                  theta_dot ** 2) / self.mass_total

        # Update acceleration
        theta_acc = ((self.gravity * sin_theta - cos_theta * common) /
                     (4.0 * self.length / 3.0 - self.mass_length_pole *
                     cos_theta ** 2 / self.mass_total))
        x_acc = (common - self.mass_length_pole * cos_theta * theta_acc /
                 self.mass_total)

        # Update velocity
        x_dot = x_dot + x_acc * self.tau
        theta_dot = theta_dot + theta_acc * self.tau

        self.state = np.array((x, x_dot, theta, theta_dot))

        done = (x < -self.max_x or x > self.max_x or
                theta < -self.max_theta or theta > self.max_theta)

        if not done:
            # Still going strong!
            reward = 1.0
        else:
            # Pole has fallen!
            self.done = False
            reward = 0.0

        return self.state, reward, done

    def reset(self):
        """
        Resets the enviroment. Should be called after every episode of a
        learning procedure.

        Returns:
        - self.state: np.array, represents the current state of the
                      environment (after reset).
        """

        self.state = np.random.uniform(-0.05, 0.05, size=4)
        self.done = False
        return self.state

    def render_state(self, state):
        """
        Renders an environment state visually. Currently shows every state
        separately (i.e. no animation).

        Arguments:
        - state: np.array, represents the current state of the
               environment.
        """
        # TODO: Make this do an actual animation
        x, _, theta, _ = state
        sin_theta, cos_theta = math.sin(theta), math.cos(theta)

        # Set cart height
        y = self.length / 2

        # Top of pole
        x_pole_top = x + 2 * self.length * sin_theta
        y_pole_top = y + 2 * self.length * cos_theta

        plt.plot([x, x_pole_top], [y, y_pole_top])
        plt.ylim(0, y + 2.5 * self.length)
        plt.xlim(- self.length, self.length)
        plt.show(block=False)
        plt.pause(0.08)
        plt.close()
