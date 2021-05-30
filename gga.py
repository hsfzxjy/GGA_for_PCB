import pygad

import copy
import numpy
import numpy as np
import functools
from typing import Tuple

class GGA(pygad.GA):
    """
    A modification GA to support:
    - ATC, WRC crossover
    - Inversion, Shuffle mutation
    - Survey Stage & Evolution Stage
    - \alpha-survive

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

    def __init__(self, *args, **kwargs):
        def add_attr(name: str):
            setattr(self, name, kwargs.get(name))
            if name in kwargs: del kwargs[name]
        add_attr("problem_ins")
        add_attr("solution_cls")
        add_attr("ggaparameters")
        add_attr("gga_crossover_type")
        return super().__init__(*args, **kwargs)


    def gga_atc_crossover(self, parent: np.ndarray):
        """
        Asexual transposition crossover (ATC) for binary sequence crossover
        """
        def atc(b):
            newb = b.copy()
            gene_start = np.random.randint(b.shape[0])
            g1, g2 = b[gene_start-2], b[gene_start-1]
            # to find the next [g1, g2] after random_ind
            bb = np.concatenate([b[gene_start:], b[:gene_start]])
            # TODO: match [~g1, ~g2]
            matching = (bb[:-1] == g1) & (bb[1:] == g2)
            # no enough transposons, cannot find insert_point
            if sum(matching) < 2: return newb
            gene_end = 1 + gene_start + matching.argmax() # fisrt true
            insert_point = gene_start + matching.argmax() + 2 + matching[matching.argmax()+1:].argmax() # second true
            shift = insert_point - gene_end
            newb[np.arange(gene_start+shift,gene_end+shift+1)%newb.shape[0]] = b[np.arange(gene_start,gene_end+1)%b.shape[0]]
            newb[np.arange(gene_start,gene_start+shift)%newb.shape[0]] = b[np.arange(gene_end+1,gene_end+1+shift)%b.shape[0]]
            return newb

        b = parent.x.T.reshape(-1)
        nb = atc(b)
        # check & fix nb
        sum_nb = nb.reshape((self.problem_ins.m, self.problem_ins.n)).sum(axis=0)
        assign = nb.reshape((self.problem_ins.m, self.problem_ins.n)).argmax(axis=0)

        x = np.zeros((self.problem_ins.n, self.problem_ins.m), dtype=np.int32)
        for i in range(self.problem_ins.n):
            if sum_nb[i] == 0: x[i, np.random.randint(0,self.problem_ins.m)] = 1
            else: x[i, assign[i]] = 1

        offspring = copy.deepcopy(parent)
        offspring.x = x
        return offspring


    def gga_wrc_crossover(self, p1: np.ndarray, p2: np.ndarray):
        # TODO: bottleneck, to be optimized
        def neibor(sol, k):
            """
            return components need vector of prev_job, job, next_job
            return all-zero vector when missing
            """
            job = self.problem_ins.A[:,k].reshape(-1)
            machine_id = sol.x[k,:].argmax()
            same_machine = (sol.x[:,machine_id] == 1)
            orders = sol.order[same_machine]
            next_order = orders[orders > sol.order[k]]
            prev_order = orders[orders < sol.order[k]]
            if next_order.shape[0]:
                next_jobid = np.where(sol.order == next_order.min())[0]
                next_job = self.problem_ins.A[:,next_jobid].reshape(-1)
            else:
                next_job = np.array([0] * self.problem_ins.p)
            if prev_order.shape[0]:
                prev_jobid = np.where(sol.order == prev_order.max())[0]
                prev_job = self.problem_ins.A[:,prev_jobid].reshape(-1)
            else:
                prev_job = np.array([0] * self.problem_ins.p)

            return prev_job, job, next_job

        def weight(sol, k):
            """
            return weight (loading & unloading) of job k
            """
            prev_job, job, next_job = neibor(sol, k)
            w1 = job.sum() - (prev_job & job).sum()
            w2 = next_job.sum() - (next_job & job).sum()
            return w1 + w2

        w = np.array([weight(p1, k) + weight(p2, k) for k in range(problem.n)])
        ind = w.argsort()

        # select one parent as workload template
        p = np.random.choice([p1, p2])
        x = np.zeros((self.problem_ins.n, self.problem_ins.m), dtype=np.int32)

        sizes = p.x.sum(axis=0)
        bins = np.split(ind, np.cumsum(sizes)[:-1])
        for t in range(len(bins)):
            x[bins[t],t] = 1
        order = np.zeros((self.problem_ins.n,), dtype=np.int32)
        order[ind] = range(self.problem_ins.n)

        offspring = self.solution_cls(problem=self.problem_ins, x=x, order=order)
        return offspring


    def gga_crossover(self, parents: np.ndarray, offspring_size: Tuple[int, int], enable_atc=False, enable_wrc=True):
        assert enable_atc or enable_wrc
        parents = list(map(lambda gene: self.solution_cls.decode(self.problem_ins, gene), parents))

        offsprings = []

        for i in range(len(parents))[::2]:
            p1, p2 = parents[i], parents[i+1]

            if enable_wrc:
                if np.random.rand() < self.ggaparameters.Pc:
                    offsprings.append(self.gga_wrc_crossover(p1, p2))
                if len(offsprings) >= offspring_size[0]: break

            if enable_atc:
                if np.random.rand() < self.ggaparameters.Pc:
                    offsprings.append(self.gga_atc_crossover(p1))
                if len(offsprings) >= offspring_size[0]: break
                if np.random.rand() < self.ggaparameters.Pc:
                    offsprings.append(self.gga_atc_crossover(p2))
                if len(offsprings) >= offspring_size[0]: break

        offsprings = np.stack([x.encode() for x in offsprings], axis=0)
        return offsprings


    def gga_mutation(self, offsprings: np.ndarray):
        mutation_probability = self.mutation_percent_genes / 100

        offsprings = list(map(lambda gene: self.solution_cls.decode(self.problem_ins, gene), offsprings))

        for offspring in offsprings:
            # p-mutation
            if np.random.rand() > mutation_probability: continue

            # randomly choose a board, then assign it to a random machine
            board_id = np.random.randint(self.problem_ins.n)
            offspring.x[board_id, :] = 0
            offspring.x[board_id, np.random.randint(self.problem_ins.m)] = 1

            # randomly choose a machine, then shuffle the orders of assigned boards
            machine_id = np.random.randint(self.problem_ins.m)
            shuflle_idx = (offspring.x.argmax(axis=1) == machine_id)
            offspring.order[shuflle_idx] = np.random.permutation(offspring.order[shuflle_idx])

        offsprings = np.stack([x.encode() for x in offsprings], axis=0)
        return offsprings


    def gga_run(self):

        """
        Guided Genetic Algorithms
        Runs the genetic algorithm. This is the main method in which the genetic algorithm is evolved through a number of generations.
        """


        self.mutation = self.gga_mutation
        if self.gga_crossover_type == "atc":
            self.crossover = functools.partial(self.gga_crossover, enable_atc=True, enable_wrc=False)
        if self.gga_crossover_type == "atc_wrc":
            self.crossover = functools.partial(self.gga_crossover, enable_atc=True, enable_wrc=True)

        if self.crossover is None:
            self.crossover = self.gga_crossover

        self.select_parents = self.roulette_wheel_selection

        # GGA: survive_ratio = \alpha when not in SURVEY stage
        if self.keep_parents == 1:
            survive_ratio = self.ggaparameters.alpha
        else:
            survive_ratio = 0

        if self.valid_parameters == False:
            raise ValueError("ERROR calling the run() method: \nThe run() method cannot be executed with invalid parameters. Please check the parameters passed while creating an instance of the GA class.\n")

        # Measuring the fitness of each chromosome in the population. Save the fitness in the last_generation_fitness attribute.
        self.last_generation_fitness = self.cal_pop_fitness()

        for generation in range(self.num_generations):
            best_solution, best_solution_fitness, best_match_idx = self.best_solution(pop_fitness=self.last_generation_fitness)

            # Appending the fitness value of the best solution in the current generation to the best_solutions_fitness attribute.
            self.best_solutions_fitness.append(best_solution_fitness)

            # Appending the best solution to the best_solutions list.
            if self.save_best_solutions:
                self.best_solutions.append(best_solution)

            # GGA: the top \alpha parents will survive
            n_survive = int(survive_ratio * self.num_parents_mating)
            n_children = self.num_parents_mating - n_survive
            self.last_survive = self.rank_selection(self.last_generation_fitness, num_parents=n_survive)

            # GGA: Selecting the best parents in the population for mating.
            #      These selected parents are expected to generate n_children children
            # TODO: in the rest, not in the original population
            n_matting = int(2 / self.ggaparameters.Pc * n_children) # 2 parents -> ~0.8 child
            n_matting += n_matting % 2
            self.last_generation_parents = self.select_parents(self.last_generation_fitness, num_parents=n_matting)

            # GGA: Generating offspring using crossover.
            self.last_generation_offspring_crossover = self.crossover(self.last_generation_parents,
                                                    offspring_size=(n_children, self.num_genes))

            # Adding some variations to the offspring using mutation.
            self.last_generation_offspring_mutation = self.mutation(self.last_generation_offspring_crossover)

            ## GGA: new pop = offsprings + last_survive
            self.population = numpy.concatenate((self.last_generation_offspring_mutation, self.last_survive), axis=0)

            self.generations_completed = generation + 1 # The generations_completed attribute holds the number of the last completed generation.

            # Measuring the fitness of each chromosome in the population. Save the fitness in the last_generation_fitness attribute.
            self.last_generation_fitness = self.cal_pop_fitness()

        # Save the fitness value of the best solution.
        _, best_solution_fitness, _ = self.best_solution(pop_fitness=self.last_generation_fitness)
        self.best_solutions_fitness.append(best_solution_fitness)

        self.best_solution_generation = numpy.where(numpy.array(self.best_solutions_fitness) == numpy.max(numpy.array(self.best_solutions_fitness)))[0][0]
        # After the run() method completes, the run_completed flag is changed from False to True.
        self.run_completed = True # Set to True only after the run() method completes gracefully.

        if not (self.on_stop is None):
            self.on_stop(self, self.last_generation_fitness)

        # Converting the 'best_solutions' list into a NumPy array.
        self.best_solutions = numpy.array(self.best_solutions)
