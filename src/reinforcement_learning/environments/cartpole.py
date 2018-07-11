"""
CartPole environment for reinforcement learning.
"""

import math
import numpy as np


class CartPoleEnv(object):
    def __init__(self):
        self.gravity = 9.81
        self.mass_cart = 1.0
        self.mass_pole = 0.1
        self.mass_total = self.mass_cart + self.mass_pole
        self.length = 0.5  # half pole length
        self.mass_length_pole = self.mass_pole * self.length
        self.force = 10  # strength of movement TODO: make learnable?
        self.tau = 0.01  # timestep of simulation

        # Failure conditions
        self.max_angle = 12 * 2 * math.pi / 360
        self.max_x = 2.4

        # Initialise environment for learning
        self.state = self.reset()

        # Define action and state spaces for learners to interact with
        self.action_space = {'left': 0, 'right': 1}
        self.state_space = {'x': type(state[0]),
                            'x_dot': type(state[1]),
                            'theta': type(state[2]),
                            'theta_dot': type(state[3])}

    def set_seed(self, seed=None):
        if seed is None:
            seed = 0
        self.seed = np.random.seed(seed)

    def get_seed(self):
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
        self.state = np.random.uniform(-0.05, 0.05, size=4)
        self.done = False
        return self.state
