"""
FrozenLake environment for reinforcement learning.
"""

import numpy as np

from .base_environment import BaseEnv


class FrozenLakeEnv(BaseEnv):
    def __init__(self):
        """
        Initialise the FrozenLake environment.
        """
        super().__init__()

        self.grid = [['S', 'F', 'F', 'F'],
                     ['F', 'H', 'F', 'H'],
                     ['F', 'F', 'F', 'H'],
                     ['H', 'F', 'F', 'G']]

        self.env_info = {'policy_type': 'grid',
                         'action_space': [0, 1, 2, 3],
                         'state_size': 16}
        self.max_reward_per_episode = 1.0

    def reset(self):
        """
        Resets the enviroment. Should be called after every episode of a
        learning procedure.

        Returns:
        - self.state: np.array, represents the current state of the
                      environment (after reset).
        """
        self.state = 0
        self.done = False
        return self.state

    def step(self, action):
        """
        Do a step in the simulation. Requires an action to be taken in
        (0, 1, 2, 3).

        The state is fully defined by the current position in the grid.

        Arguments:
        - action: one of (0, 1, 2, 3), represents the action to take (up,
                  right, down, left).

        Returns:
        - self.state: np.array, represents the current state of the
                      environment.
        - reward: float, the reward earned during this step.
        - done: bool, whether the episode should finish after this step (i.e.
                the pole has dropped too far).
        """
        assert action in (0, 1, 2, 3), ("action must be 0 (up), 1 (right), "
                                        "2 (down), or 3 (left); "
                                        "not {}".format(action))

        # Check if already finished
        if self.done:
            print('WARNING: calling step() even though the environment has '
                  'already returned done=True. Call reset() first before '
                  'step() again: anything beyond this is undefined behaviour.')

        # Get current state
        state = self.state
        row = state % 4
        col = state // 4

        print(state, self.grid[row][col], action)
        # Do a step on the grid as given by action
        # TODO: Add slipperyness
        if action == 0:  # up
            if row != 0:
                row -= 1
        elif action == 1:  # right
            if col != len(self.grid[0]) - 1:
                col += 1
        elif action == 2:  # down
            if row != len(self.grid) - 1:
                row += 1
        elif action == 3:  # left
            if col != 0:
                col -= 1

        self.state = len(self.grid) * row + col

        # Reached goal?
        if self.grid[row][col] == 'G':
            return self.state, 1.0, True

        # Fallen into hole?
        done = (self.grid[row][col] == 'H')

        # Reward only 1.0 if goal is reached
        reward = 0.0

        return self.state, reward, done
