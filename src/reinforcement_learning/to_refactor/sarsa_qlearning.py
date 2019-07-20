import numpy as np
import random


class CliffWalk(object):
    def __init__(self, height=4, width=12, timesteps=100):
        self.height = height
        self.width = width
        self.timesteps = timesteps

        # up, down, left, right
        self.actions = list(range(4))
        # left to right, bottom to top
        self.states = list(range(height * width))
        self.cliff = list(range(1, width - 1))

        self.state = 0
        self.value_func = np.zeros(len(self.states))

    def reset(self):
        self.__init__()

    def step(self, state, action):
        episode_end = False

        if action == 0:  # up
            if state >= (self.height - 1) * self.width:  # can't go up
                new_state = state
            else:
                new_state = state + self.width

        elif action == 1:  # down
            if state < self.width:  # can't go down
                new_state = state
            else:
                new_state = state - self.width

        elif action == 2:  # left
            if state % self.width == 0:  # can't go left
                new_state = state
            else:
                new_state = state - 1

        elif action == 3:  # right
            if state % self.width == self.width - 1:  # can't go right
                new_state = state
            else:
                new_state = state + 1

        if new_state in self.cliff:
            reward = -100
            new_state = 0
        elif new_state == self.width - 1:
            reward = 0
            episode_end = True
        else:
            reward = -1

        return reward, new_state, episode_end

    def simulate(self, policy, verbose=False):
        state = self.state
        total_reward = 0
        for _ in range(self.timesteps):
            action = policy[state]
            if verbose:
                print(state, action)
            reward, state, episode_end = self.step(state, action)
            if verbose:
                print(state, reward)
                print('\n')
            total_reward += reward
            if episode_end:
                break
        return total_reward


class Agent(object):
    def __init__(self, env, num_episodes=1000):
        self.env = env
        self.num_episodes = num_episodes
        self.alpha = 0.1
        self.gamma = 0.9

    def random_policy(self):
        policy = np.random.choice(self.env.actions, size=len(self.env.states))
        total_reward = self.env.simulate(policy)

    def eps_greedy_q(self, av_func, state, t):
        # Selects epsilon greedy action from av_func given current state,
        # where epsilon equals 1 over the current episode number (converges
        # to greedy policy).
        action = np.argmax(av_func[state, :])  # greedy action
        if random.uniform(0, 1) < 1 / (t + 1):  # make epsilon-greedy
            action = random.choice(self.env.actions)
        return action

    def sarsa(self):
        self.env.reset()
        # Initialise action-value function (num_states x num_actions)
        av_func = np.zeros((len(self.env.states), len(self.env.actions)))
        for t in range(self.num_episodes):
            # Initialise state
            state = env.state
            # Get action for this state from eps-greedy policy following Q (state-value function)
            action = self.eps_greedy_q(av_func, state, t)
            # Run the episode until termination
            while True:
                # Do step in environment
                reward, new_state, end = env.step(state, action)
                # Get new action from eps-greedy policy following Q
                new_action = self.eps_greedy_q(av_func, new_state, t)
                # Do state-value function update (SARSA)
                av_func[state, action] += self.alpha * (reward + self.gamma * av_func[new_state, new_action] - av_func[state, action])
                if end:  # termination
                    break
                # Update state and action for next step
                state = new_state
                action = new_action
        return av_func

    def qlearning(self):
        self.env.reset()
        # Initialise action-value function (num_states x num_actions)
        av_func = np.random.normal(size=(len(self.env.states), len(self.env.actions)))
        for t in range(self.num_episodes):
            # Initialise state
            state = env.state
            # Run the episode until termination
            while True:
                # Get action for this state from eps-greedy policy following Q (state-value function)
                # We put this inside the loop now, because we will need a new action for every timestep
                # in the simulation: Q-learning does not work on-policy, so we do not get new actions after
                # every step: instead we maximise over actions in the update step, and randomly pick a
                # new off-policy action for the next step (in the new state).
                action = self.eps_greedy_q(av_func, state, t=20)
                # Do step in environment
                reward, new_state, end = env.step(state, action)
                # Do state-value function update (SARSA)
                av_func[state, action] += self.alpha * (reward + self.gamma * np.max(av_func[new_state, :]) - av_func[state, action])
                if end:  # termination
                    break
                # Update state and action for next step
                state = new_state
        return av_func


if __name__ == "__main__":
    env = CliffWalk()
    agent = Agent(env)

    # av_func = agent.sarsa()
    av_func = agent.qlearning()
    print("Optimal action-value function: \n {}".format(av_func))

    policy = np.argmax(av_func, axis=1)
    total_reward = env.simulate(policy, verbose=True)
    print("Total reward under this policy: {}".format(total_reward))
