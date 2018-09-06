"""
Agent that learns by evolution search.

Based on blog post by Moustafa Alzantot:
https://becominghuman.ai/genetic-algorithm-for-reinforcement-learning-a38a5612c4dc
"""

import sys
import itertools
import numpy as np
from base_learner import BaseLearner
from random_search import RandomSearch

sys.path.append("..")  # noqa: F401
from environments.cartpole import CartPoleEnv
from environments.frozenlake import FrozenLakeEnv


INITIAL_GEN_SIZE = 100
NUM_GENERATIONS = 10
NUM_ELITE = 10
NUM_CHILDREN_PER_PARENT_COMB = 2
MUTATION_PROB = 0.05


class EvolutionSearch(BaseLearner):
    def __init__(self, env):
        super().__init__(env)

        # Initial generation size
        self.initial_gen_size = INITIAL_GEN_SIZE
        # Number of generations to run for
        self.num_generations = NUM_GENERATIONS
        # Number of parents to keep each generation
        self.num_elite = NUM_ELITE
        # Number of children each generation
        self.num_children_per_parent_comb = NUM_CHILDREN_PER_PARENT_COMB
        # Probability of mutation
        self.mutation_prob = MUTATION_PROB

        # Use random policy generator from RandomSearch learner
        self.get_policy = RandomSearch(env).get_policy

    def crossover(self, parent1, parent2):
        """
        Generates child policy from two parent policies.

        Arguments:
        - parent1: np.array, represents the first parent policy.
        - parent2: np.array, represents the second parent policy.

        Returns:
        - child: np.array, represents the resulting child policy.
        """

        child = parent1.copy()
        for i in range(child.size):
            # Get either gene with probability 0.5
            p = np.random.uniform()
            if p < 0.5:
                child[i] = parent2[i]
        return child

    def mutate(self, child):
        """
        Mutates child genes with some probability.

        Arguments:
        - child: np.array, represents the child policy to mutate.

        Returns:
        - child: np.array, represents the mutated child policy.
        """

        # Get random genes for possible mutation
        random_child = self.get_policy()
        for i in range(child.size):
            # Mutate gene with probability mutation_prob
            p = np.random.uniform()
            if p < self.mutation_prob:
                child[i] = random_child[i]
        return child

    def evolution_step(self, policy_pop):
        """
        Performs a single step of evolution.

        The initial policy population is
        scored, and the num_elite best policies are used to form the next
        generation. For every combination of two parents in the elite set, we
        generate num_children_per_parent_comb children by randomly taking genes
        (values in the numpy array that represents the policy) from either
        parent with equal probability. Finally every child gene has a
        mutation_prob probability of mutation. This process repeats for
        num_generations generations, and the final policy population is
        returned.

        Arguments:
        - policy_pop: list of policies, representing the initial population.

        Returns:
        - policy_pop: list of policies, representing the final population.
        """
        # Evaluate parents
        parent_scores = [self.evaluate(policy, do_print=False) for policy
                         in policy_pop]

        # Get best scoring parents
        best_parents = list(np.array(policy_pop)[
            np.argsort(parent_scores)[-self.num_elite:], :])
        # For every combination of parents, generate
        # num_children_per_parent_comb children
        # End up with (num_elite! / ((num_elite - 2)! 2!) *
        # num_children_per_parent_comb) children
        all_children = []
        for parents in itertools.combinations(best_parents, 2):
            children = [self.crossover(parents[0], parents[1])
                        for _ in range(self.num_children_per_parent_comb)]
            all_children.extend(children)

        # Children have mutations as well
        mutated_children = [self.mutate(child) for child in
                            all_children]

        # Next generations
        policy_pop = best_parents + mutated_children
        return policy_pop

    def train(self, initial_gen_size=None, num_generations=None,
              num_elite=None, num_children_per_parent_comb=None,
              mutation_prob=None, max_t=None):
        """
        Train the EvolutionSearch learner. Sets self.best_policy to the best
        found policy.

        Training consists of starting with an initial randomly generated set
        of parent policies of size initial_gen_size, after which
        num_generations evolution steps are performed. The final generation is
        evaluated, and the best policy determined from this.

        Arguments:
        - initial_gen_size: int, number of random policies to start with
                            (these are the initial parents).
        - num_generations: int, number of generations to run evolution for.
        - num_elite: int, number of parents to keep each generation (these are
                     the elite set that generate the next generation).
        - num_children_per_parent_comb: int, number of children produced by
                                        every possible combination of two
                                        parents in the elite set.
        - max_t: int, number of steps in an episode.
        """

        if initial_gen_size is not None:
            self.initial_gen_size = initial_gen_size
        if num_generations is not None:
            self.num_generations = num_generations
        if num_elite is not None:
            self.num_elite = num_elite
        if num_children_per_parent_comb is not None:
            self.num_children_per_parent_comb = num_children_per_parent_comb
        if mutation_prob is not None:
            self.muation_prob = mutation_prob
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

        # Initialise population of random policies to start with
        policy_pop = [self.get_policy() for _ in range(self.initial_gen_size)]
        # Do num_generations evolution steps
        for _ in range(self.num_generations):
            policy_pop = self.evolution_step(policy_pop)

        # Determine best policy on an average of 10 episodes
        final_scores = [self.evaluate(policy, do_print=False, average=10) for
                        policy in policy_pop]
        # That policy is the winner!
        win_ind = np.argmax(final_scores)
        self.best_score = final_scores[win_ind]
        self.best_policy = policy_pop[win_ind]


if __name__ == '__main__':
    env = CartPoleEnv()
    # env = FrozenLakeEnv()
    learner = EvolutionSearch(env)

    learner.train(max_t=100)
    learner.evaluate(learner.best_policy, do_show=False, average=1000)
