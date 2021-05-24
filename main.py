from typing import Tuple
import numpy as np
import pygad
from pcb import Problem, Solution


def fitness_func(gene: np.ndarray, _: int) -> float:
    sol = Solution.decode(problem, gene)

    # pygad maximizes the fitness function, so we add a minus here
    return -sol.f


def on_generation(ga: pygad.GA):
    ...


class GA(pygad.GA):
    """
    A DIRTY modification to support customized crossover / mutation.
    
    Since pygad provides no way for customizing genetic operations, we override certain built-in
    functions to achieve the goal. 
    
    NOTE that the name of the functions below will not match their behaviors. Simply
    fill the lines marked with TODO.

    About class Solution
    ====================

    Since pygad does not support custom type as gene, we turn to use a 2n-sized array to
    represent a gene. When recieving a gene from pygad, use `Solution.decode()` to translate
    it into a Solution instance; before returning some Solutions back to pygad, use 
    `solution.encode()` to translate them back to 2n-sized arrays.

    See comments in class Solution definition for details of its members.
    """
    def single_point_crossover(self, parents: np.ndarray, offspring_size: Tuple[int, int]):
        parent1, parent2 = map(lambda gene: Solution.decode(problem, gene), parents)
        num_offspring = offspring_size[0]

        offsprings = []
        for i in range(num_offspring):
            # TODO: generate offspring from parent1 and parent2
            offspring = ...
            offsprings.append(offspring)

        offsprings = np.stack([x.encode() for x in offsprings], axis=0)
        return offsprings

        # return super().single_point_crossover(parents, offspring_size)

    def random_mutation(self, offsprings: np.ndarray):
        offsprings = list(map(lambda gene: Solution.decode(problem, gene), offsprings))
        ...  # TODO: do mutations on offsprings
        offsprings = np.stack([x.encode() for x in offsprings], axis=0)
        return offsprings

        # return super().random_mutation(offsprings)


if __name__ == "__main__":
    np.random.seed(43)
    n = 100
    m = 3
    p = 100
    A = (np.random.rand(p, n) > 0.5).astype(np.int32)
    problem = Problem(
        n=n,
        m=m,
        p=p,
        C=np.array([A.sum(axis=0).max() + 2] * m, dtype=np.int32),
        A=A,
        S=3,
        s=1,
        lambda1=0.2,
        lambda2=0.2,
        lambda3=0.6,
    )

    ga_instance = GA(
        num_generations=200,
        num_parents_mating=2,
        fitness_func=fitness_func,
        sol_per_pop=20,
        num_genes=problem.n * 2,
        init_range_low=0,
        init_range_high=1,
        parent_selection_type="sss",
        keep_parents=1,
        crossover_type="single_point",
        mutation_type="random",
        mutation_percent_genes=10,
        on_generation=on_generation,
    )
    ga_instance.run()
    ga_instance.plot_result()