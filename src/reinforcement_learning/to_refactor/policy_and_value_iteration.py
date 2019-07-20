import numpy as np
import math


class CarDealer(object):
    def __init__(self):
        # Problem parameters
        self.timesteps = 100
        # self.min_init_cars = 4
        # self.max_cars = 10
        # self.max_move = 5
        # self.exp_ret = [3, 2]
        # self.exp_req = [3, 4]

        self.min_init_cars = 3
        self.max_cars = 6
        self.max_move = 5
        self.exp_ret = [3, 2]
        self.exp_req = [3, 4]

        # State and actions
        self.actions = list(range(-self.max_move, self.max_move + 1))
        self.state = np.random.randint(self.min_init_cars, self.max_cars + 1, size=2)
        # Value of every possible state (tabular)
        self.value_func = np.zeros((self.max_cars + 1, self.max_cars + 1))

    def reset(self):
        self.__init__()

    def step(self, action, state, returned, requested):
        # Does step in environment given previous state, returned cars,
        # requested cars, and action (cars moved)

        # For consistency
        state = np.array(state)

        # Cars are returned
        state += returned
        # Cars above max_cars disappear
        state = np.min([state, [self.max_cars, self.max_cars]], axis=0)
        # Rent out all possible cars up to requested number
        rent = np.min([requested, state], axis=0)
        state -= rent

        # Reward from renting
        reward = np.sum(rent * 10)

        # Apply policy (move cars_moved from position 1 to position 2)
        cars_moved = action
        if cars_moved > 0:  # move from 1 to 2
            # Check that we don't go over max_cars location 2
            cars_moved = min(cars_moved, self.max_cars - state[1])
            # Check that we have this many cars at location 1
            cars_moved = min(cars_moved, state[0])
            state += (-cars_moved, cars_moved)
        else:  # move from 2 tot 1
            # Check that we don't go over max_cars location 1
            cars_moved = max(cars_moved, state[0] - self.max_cars)
            # Check that we have this many cars at location 2
            cars_moved = max(cars_moved, -state[1])
            state += (-cars_moved, cars_moved)
        # Could change policy after this check to satisfy constraints of problem,
        # but this doesn't seem strictly necessary, since opting to move more cars
        # than possible will not increase or decrease reward compared to moving
        # the maximum number possible.

        # Cost of moving cars
        reward -= np.abs(cars_moved) * 2
        return reward, tuple(state)

    def sample_step(self, action, state):
        # Cars are returned
        ret = np.random.poisson(self.exp_ret)
        # Requested cars are rented
        req = np.random.poisson(self.exp_req)

        # Do step in environment given sample
        reward, state = self.step(action, state, ret, req)

        return reward, state

    def expected_return(self, action, state, value_func, discount=1., mode='analytic'):
        # Calculates expected future return given action, state and value_function

        # Currently does Monte Carlo and analytic approximation of environment dynamics
        # Much faster would be to analytically approximate all state-action-state
        # transitions once, and construct the transition matrix from those values.
        # This turns the expected value step into a simple matrix multiplication.

        # Monte Carlo
        if mode == 'MC':

            total_return = 0
            for t in range(100):
                # Cars are returned
                ret = np.random.poisson(self.exp_ret)
                # Requested cars are rented
                req = np.random.poisson(self.exp_req)
                # Step in environment
                reward, next_state = self.step(action, state, ret, req)
                total_return += reward + discount * value_func[next_state]

            # Return average expected return for this state, given this action
            return total_return / (t + 1)

        # Analytic calculation with cutoff on higher Poisson modes
        elif mode == 'analytic':
            def poisson(n, l):
                return l**n / math.factorial(n) * np.exp(-l)

            def mode_cutoff(exp):
                # Returns Poisson mode at which to cut off, given expected number l
                # Values found by inspection Poisson distribution graph and seeing
                # where the probability goes to 0
                if exp == 1:
                    return 4
                elif exp == 2:
                    return 6
                elif exp == 3:
                    return 8
                elif exp == 4:
                    return 10
                elif exp == 5:
                    return 11
                else:
                    return l * 2

            exp_return = 0
            # Loop over possible configurations of return-requested
            # This is what makes the environment dynamics expensive to calculate:
            # There are many possible end states and rewards given any state-action pair.
            for i in range(mode_cutoff(self.exp_ret[0])):
                for j in range(mode_cutoff(self.exp_ret[1])):
                    for k in range(mode_cutoff(self.exp_req[0])):
                        for l in range(mode_cutoff(self.exp_req[1])):
                            # Get weight of this configuration
                            weight = poisson(i, self.exp_ret[0]) * poisson(j, self.exp_ret[1]) * poisson(k, self.exp_req[0]) * poisson(l, self.exp_req[1])
                            # Do step in simulation
                            reward, next_state = self.step(action, state, np.array([i, j]), np.array([k, l]))
                            # Calculate expected return (weighted average)
                            exp_return += (reward + discount * value_func[next_state]) * weight
            return exp_return
        else:
            raise ValueError("'mode' should be 'MC' (Monte Carlo) or 'analytic', not {}".format(mode))

    def simulate(self, agent):
        self.reset()
        self.total_reward = 0
        for t in range(self.timesteps):
            reward, state = self.sample_step(agent.policy[self.state], self.state)
            self.state = state
            self.total_reward += reward


class Agent(object):
    def __init__(self, env):
        # Action for every possible state (tabular)
        self.env = env
        self.discount = 0.9
        self.convergence = 5.
        # self.policy = np.random.choice(env.actions, size=env.value_func.shape)
        self.policy = np.zeros(env.value_func.shape, dtype='int32')

    def iterative_policy_evaluation(self, policy, value_func):
        # Evaluate the given policy (evalute state value function for this policy)
        # Does Iterative Policy Evaluation
        print("- Evaluating policy...\n  {}".format(policy))
        while True:
            delta = 0
            old_value_func = value_func
            # Sweep over all states
            for i in range(value_func.shape[0]):
                for j in range(value_func.shape[1]):
                    # Cache state value
                    state = (i, j)
                    value = old_value_func[state]
                    # Find action to take according to the policy
                    policy_action = policy[state]
                    # Get expected return given this action in this state, with the current value function
                    expected_return = env.expected_return(policy_action, state, old_value_func, self.discount)
                    # Update value function with expected return
                    value_func[state] = expected_return
                    # Update error
                    delta = max(delta, np.abs(expected_return - value))

            # Check convergence
            print('    Delta: {:.2f}'.format(delta))
            if delta < self.convergence:
                break
        return value_func

    def get_greedy_action(self, state, value_func):
        # Given a state and value function, determine the greedy action
        # Environment dynamics are contained in self.env.step()
        max_return = -1e9
        for action in env.actions:
            # Get expected return given this action
            expected_return = self.env.expected_return(action, state, value_func, self.discount)
            # Keep action with highest expected return
            if expected_return > max_return:
                max_return = expected_return
                greedy_action = action
        return greedy_action

    def policy_improvement(self, policy, value_func):
        # Does policy improvement (takes greedy action given converged value function)
        print("- Improving policy...")
        policy_stable = True
        # Sweep over all states
        for i in range(value_func.shape[0]):
            for j in range(value_func.shape[1]):
                # Cache action according to current policy
                state = (i, j)
                policy_action = policy[state]
                # Find greedy action given value function
                greedy_action = self.get_greedy_action(state, value_func)
                # Update policy
                policy[state] = greedy_action
                # Check convergence: True if policy doesn't change for all actions
                if policy_action != greedy_action:
                    policy_stable = False

        return policy, policy_stable

    def policy_iteration(self):
        self.env.reset()
        value_func = self.env.value_func
        policy = self.policy

        k = 0
        while True:
            k += 1
            print("Doing policy iteration, step {}".format(k))
            # Iterative Policy Evaluation until value function convergence
            value_func = self.iterative_policy_evaluation(policy, value_func)
            # Policy improvement step
            policy, policy_stable = self.policy_improvement(policy, value_func)
            # Stop if policy hasn't changed in the last iteration
            if policy_stable:
                break

        self.policy = policy
        self.env.value_func = value_func

        return policy, value_func

    def value_iteration(self, mode='bellman'):
        self.env.reset()
        value_func = self.env.value_func
        policy = self.policy

        if mode == 'interleaf':
            # What follows is an explicit coding of value iteration viewed as
            # 'a single step in iterative policy evaluation, followed by a
            # policy improvement step'.
            k = 0
            while True:
                k += 1
                print("Doing value iteration, step {}".format(k))

                # One step of value iteration...
                old_value_func = value_func
                # Sweep over all states (could even do in-place: i.e. update
                # V(s) for a single state, then do policy improvement, repeat).
                for i in range(value_func.shape[0]):
                    for j in range(value_func.shape[1]):
                        # Cache state value
                        state = (i, j)
                        value = old_value_func[state]
                        # Find action to take according to the policy
                        policy_action = policy[state]
                        # Get expected return given this action in this state, with the current value function
                        expected_return = env.expected_return(policy_action, state, old_value_func, self.discount)
                        # Update value function with expected return
                        value_func[state] = expected_return

                # ... followed by one step of policy improvement
                policy, policy_stable = self.policy_improvement(policy, value_func)

                if policy_stable:
                    break

        elif mode == 'bellman':
            # Alternatively, an interpretation of value iteration as a fixed point
            # equation for the Bellman Optimality Equation.
            k = 0
            while True:
                k += 1
                print("Doing value iteration, step {}".format(k))
                delta = 0
                old_value_func = value_func
                # Sweep over all states (could even do in-place: i.e. update
                # V(s) for a single state, then do policy improvement, repeat).
                for i in range(value_func.shape[0]):
                    for j in range(value_func.shape[1]):
                        # Cache state value
                        state = (i, j)
                        value = old_value_func[state]

                        max_return = -10e10
                        # Loop over actions to find best action
                        for action in env.actions:
                            # Get expected return given this action in this state, with the current value function
                            expected_return = env.expected_return(action, state, old_value_func, self.discount)
                            if expected_return > max_return:
                                max_return = expected_return
                                best_action = action
                        policy[state] = best_action
                        value_func[state] = expected_return

                        # Update error
                        delta = max(delta, np.abs(max_return - value))

                # Check convergence
                print('    Delta: {:.2f}'.format(delta))
                if delta < self.convergence:
                    break

        return policy, value_func


if __name__ == "__main__":
    env = CarDealer()
    agent = Agent(env)

    # policy, value_function = agent.policy_iteration()
    policy, value_function = agent.value_iteration()
    print("\n\nOptimal policy:\n  {}".format(policy))
    print("\nOptimal value function:\n  {}".format(value_function))
