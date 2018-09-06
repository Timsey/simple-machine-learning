"""
Base environment on which the reinforcement learning environments are based.
"""

class BaseEnv(object):
    def __init__(self):
        """
        Initialise base environment.

        self.env info should be a dictionary containg the information
        necessary for the learners to interact with the environments.
        - Example for Cart Pole (binary separating hyperplane):
            self.env_info = {'policy_type': 'binary',
                             'action_space':  [0, 1],
                             'state_size': 4}
        - Example for Frozen Lake (4x4 grid):
            self.env_info = {'policy_type': 'grid',
                             'action_space': [1, 2, 3, 4],
                             'state_size': 16}
        - Example for (e.g.) Pong through image instead of specific features:
            self.env_info = {'policy_type': 'function',
                             'action_space': [0, 1],
                             'state_size': IMAGE_DIMENSION}

        Either of the following need to be set in the environment, the other
        should be None:
        - self.max_reward_per_episode is the maximum reward that can be reached
            in an episode. Used to evaluate learners. Default None.
        - self.max_reward_per_timestep is the maximum reward that can be reached
            in a time step. Used to evaluate learners. Default None.
        """

        # Define the info the learners need to interact with the environment
        self.env_info = {}  # Set this in environment __init__
        self.max_reward_per_episode = None  # Set either of these two
        self.max_reward_per_timestep = None  # Set either of these two

        # Call self.reset here after it has been implemented in the actual
        # environment.

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

    # -----------------
    # Implement the functions below when writing an environment
    # -----------------
    def reset(self):
        """
        Resets the enviroment. Should be called after every episode of a
        learning procedure. Function needs to be completed in the actual
        environment by setting self.state.

        Returns:
        - self.state: np.array, represents the current state of the
                      environment (after reset).
        """
        # set this to an actual state in the environment reset function
        self.state = None

        self.done = False
        return self.state


    def step(self, action):
        """
        Do a step in the simulation. The step() function of other environments
        should include at least this code.

        Arguments:
        - action: action to take.

        Returns:
        - self.state: np.array, represents the current state of the
                      environment.
        - reward: float, the reward earned during this step.
        - done: bool, whether the episode should finish after this step (i.e.
                the pole has dropped too far).
        """

        # Check if already finished
        if self.done:
            print('WARNING: calling step() even though the environment has '
                  'already returned done=True. Call reset() first before '
                  'step() again: anything beyond this is undefined behaviour.')

        # Get current state
        state = self.state

        # ------
        # DO STEP OF SIMULATION HERE
        # I.e. update state given current state and action
        # ------

        done = None  # set to true on failure condition or on episode end
        reward = None  # set this to actual reward depending on state

        return self.state, reward, done
