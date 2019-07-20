import numpy as np


class RandomWalk(object):
    def __init__(self, num_states=5):
        self.actions = [-1, 1]
        self.states = list(range(num_states + 2))  # 2 terminal states
        self.state = num_states // 2 + 1 if num_states % 2 == 0 else num_states // 2 + 2  # starting state
        self.value_func = np.zeros(len(self.states))

    def reset(self):
        self.__init__()

    def step(self, state, action):
        state += action
        if state == self.states[0]:
            reward = 0
            terminate = True
        elif state == self.states[-1]:
            reward = 1
            terminate = True
        else:
            reward = 0
            terminate = False

        return state, reward, terminate


class Agent(object):
    def __init__(self, env):
        self.env = env
        self.alpha = 0.1

    def temporal_difference(self, num_episodes=1000):
        print("Doing on-line TD(0)...")
        self.env.reset()
        value_func = env.value_func
        for _ in range(num_episodes):
            state = env.state
            terminate = False
            while not terminate:
                # Get action
                action = np.random.choice(env.actions)  # random policy
                # Do step in environment
                new_state, reward, terminate = env.step(state, action)
                # Calculate new value function estimate for this state
                new_value = value_func[state] + self.alpha * (reward + value_func[new_state] - value_func[state])
                # Update state and value function
                value_func[state] = new_value
                state = new_state
        return value_func

    # Numerically unstable? Updates explode...
    def batch_temporal_difference(self, num_episodes=10000):
        print("Doing batch TD(0)...")
        self.env.reset()
        history_dict = {key: [] for key in env.states}
        # Generate history
        print("Generating history...")
        for _ in range(num_episodes):
            state = env.state
            terminate = False
            while not terminate:
                action = np.random.choice(env.actions)
                new_state, reward, terminate = env.step(state, action)
                history_dict[state].append((new_state, reward))
                state = new_state

        print("Estimating value function...")
        # Update value function (should be easy to vectorise?)
        value_func = env.value_func
        convergence = 0.001
        k = 0
        while True:
            k += 1
            if k % 100 == 0:
                print(" iteration {}, delta: {:.4f}".format(k, delta))
            delta = 0
            updates = {key: 0 for key in history_dict.keys()}
            # Sweep over states
            for state, result in history_dict.items():
                # Sweep over history for that state
                for new_state, reward in result:
                    updates[state] += self.alpha * (reward + value_func[new_state] - value_func[state])

            # Update value function with total update
            for state in history_dict.keys():
                if len(history_dict[state]) == 0:
                    continue  # terminal states
                value_func[state] += updates[state] / len(history_dict[state])
                delta = max(delta, updates[state])

            # Check convergence
            if delta < convergence:
                print("\nConverged after {} iterations with delta: {:.4f}".format(k, delta))
                break

        return value_func


if __name__ == "__main__":
    num_states = 5

    env = RandomWalk(num_states)
    agent = Agent(env)

    # value_func = agent.temporal_difference()
    value_func = agent.batch_temporal_difference()
    print("\nEvaluated value function:\n {}".format(value_func))

    true_value_func = np.array([i / (num_states + 1) for i in range(num_states + 1)] + [0])
    print("True value function:\n {}".format(true_value_func))

    print("\nValue function MSE: {:.3f}".format(np.sum(value_func - true_value_func)**2))
