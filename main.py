from typing import Tuple
import numpy as np
import pygad
from pcb import Problem, Solution
import dataclasses
import functools
import copy

@dataclasses.dataclass
class GGAParameters:
    # Survey population size
    Ps = 50
    # Evolution population size
    Pe = 20
    # Probability of mutation
    Pm = .04
    # Probability of crossover
    Pc = .8
    # Number of survey generations
    Gs = 20
    # Number of evolution generations
    # Stopping condition: when the number of iterations reaches Ge
    # Ge = 50
    Ge = 160

    # will be selected based on the experiments, [0,1], 0.3
    alpha = 0.4

ggaparameters = GGAParameters()


def fitness_func(gene: np.ndarray, _: int) -> float:
    sol = Solution.decode(problem, gene)

    # pygad maximizes the fitness function, so we add a minus here
    return -sol.f

class GGA(pygad.GA):
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

    def __init__(self, *args, **kwargs):
        self.gga_crossover_type = kwargs.get("gga_crossover_type")
        if kwargs.get("gga_crossover_type"): del kwargs["gga_crossover_type"]
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
        sum_nb = nb.reshape((problem.m, problem.n)).sum(axis=0)
        assign = nb.reshape((problem.m, problem.n)).argmax(axis=0)

        x = np.zeros((problem.n, problem.m), dtype=np.int32)
        for i in range(problem.n):
            if sum_nb[i] == 0: x[i, np.random.randint(0,problem.m)] = 1
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
            job = sol.problem.A[:,k].reshape(-1)
            machine_id = sol.x[k,:].argmax()
            same_machine = (sol.x[:,machine_id] == 1)
            orders = sol.order[same_machine]
            next_order = orders[orders > sol.order[k]]
            prev_order = orders[orders < sol.order[k]]
            if next_order.shape[0]:
                next_jobid = np.where(sol.order == next_order.min())[0]
                next_job = sol.problem.A[:,next_jobid].reshape(-1)
            else:
                next_job = np.array([0] * sol.problem.p)
            if prev_order.shape[0]:
                prev_jobid = np.where(sol.order == prev_order.max())[0]
                prev_job = sol.problem.A[:,prev_jobid].reshape(-1)
            else:
                prev_job = np.array([0] * sol.problem.p)

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

        # select ONE parent as workload template
        p = np.random.choice([p1, p2])
        x = np.zeros((problem.n, problem.m), dtype=np.int32)

        sizes = p.x.sum(axis=0)
        bins = np.split(ind, np.cumsum(sizes)[:-1])
        for t in range(len(bins)):
            x[bins[t],t] = 1
        order = np.zeros((problem.n,), dtype=np.int32)
        order[ind] = range(problem.n)

        offspring = Solution(problem=p.problem, x=x, order=order)
        return offspring


    def gga_crossover(self, parents: np.ndarray, offspring_size: Tuple[int, int], enable_atc=False, enable_wrc=True):
        assert enable_atc or enable_wrc
        parents = list(map(lambda gene: Solution.decode(problem, gene), parents))

        offsprings = []

        for i in range(len(parents))[::2]:
            p1, p2 = parents[i], parents[i+1]

            if enable_wrc:
                if np.random.rand() < ggaparameters.Pc:
                    offsprings.append(self.gga_wrc_crossover(p1, p2))
                if len(offsprings) >= offspring_size[0]: break

            if enable_atc:
                if np.random.rand() < ggaparameters.Pc:
                    offsprings.append(self.gga_atc_crossover(p1))
                if len(offsprings) >= offspring_size[0]: break
                if np.random.rand() < ggaparameters.Pc:
                    offsprings.append(self.gga_atc_crossover(p2))
                if len(offsprings) >= offspring_size[0]: break

        offsprings = np.stack([x.encode() for x in offsprings], axis=0)
        return offsprings

        # return super().single_point_crossover(parents, offspring_size)

    def gga_mutation(self, offsprings: np.ndarray):
        mutation_probability = self.mutation_percent_genes / 100

        offsprings = list(map(lambda gene: Solution.decode(problem, gene), offsprings))

        for offspring in offsprings:
            # p-mutation
            if np.random.rand() > mutation_probability: continue

            # randomly choose a board, then assign it to a random machine
            board_id = np.random.randint(problem.n)
            offspring.x[board_id, :] = 0
            offspring.x[board_id, np.random.randint(problem.m)] = 1

            # randomly choose a machine, then shuffle the orders of assigned boards
            machine_id = np.random.randint(problem.m)
            shuflle_idx = (offspring.x.argmax(axis=1) == machine_id)
            offspring.order[shuflle_idx] = np.random.permutation(offspring.order[shuflle_idx])

        offsprings = np.stack([x.encode() for x in offsprings], axis=0)
        return offsprings

    def gga_run(self):

        """
        Guided Genetic Algorithms
        Runs the genetic algorithm. This is the main method in which the genetic algorithm is evolved through a number of generations.
        """

        import time
        import numpy

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
            survive_ratio = ggaparameters.alpha
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
            n_matting = int(2 / ggaparameters.Pc * n_children) # 2 parents -> ~0.8 child
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

if __name__ == "__main__":
    np.random.seed(43)
    ns = []
    result1 = []
    result2 = []
    result3 = []
    for n in np.arange(20, 101, 10):
        ns.append(int(n))
        n = int(n)
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

        # Survey
        ga_survey_instance = GGA(
            num_generations=ggaparameters.Gs,
            num_parents_mating=ggaparameters.Ps,
            fitness_func=fitness_func,
            sol_per_pop=ggaparameters.Ps,
            num_genes=problem.n * 2,
            init_range_low=0,
            init_range_high=1,
            keep_parents=0,
            mutation_percent_genes=ggaparameters.Pm * 100,
            save_best_solutions=False,
        )
        ga_survey_instance.gga_run()

        # select best Pe from survey result
        Pe = ggaparameters.Pe
        survey_pop = ga_survey_instance.population
        survey_fit = ga_survey_instance.last_generation_fitness
        init_pop = survey_pop[survey_fit.argsort()[::-1][:Pe]]

        ga_instance = GGA(
            num_generations=ggaparameters.Ge,
            num_parents_mating=ggaparameters.Pe,
            fitness_func=fitness_func,
            initial_population=init_pop,
            num_genes=problem.n * 2,
            init_range_low=0,
            init_range_high=1,
            keep_parents=1,
            mutation_percent_genes=ggaparameters.Pm * 100,
            save_best_solutions=True,
        )
        ga_instance.gga_run()


        best_sol_genid = ga_instance.best_solution_generation
        best_gene = ga_instance.best_solutions[best_sol_genid]
        solution = Solution.decode(problem, best_gene)
        print(solution.f3)
        print(solution.QI)
        result1.append(solution.QI)

        # GGA2
        ga_spc = GGA(
            num_generations=ggaparameters.Ge,
            num_parents_mating=ggaparameters.Pe,
            fitness_func=fitness_func,
            initial_population=init_pop,
            num_genes=problem.n * 2,
            init_range_low=0,
            init_range_high=1,
            keep_parents=0,
            mutation_percent_genes=ggaparameters.Pm * 100,
            save_best_solutions=True,
            gga_crossover_type="atc"
        )
        ga_spc.gga_run()

        best_sol_genid = ga_spc.best_solution_generation
        best_gene = ga_spc.best_solutions[best_sol_genid]
        solution = Solution.decode(problem, best_gene)
        print(solution.f3)
        print(solution.QI)
        result2.append(solution.QI)

        # GGA3
        ga_mpc = GGA(
            num_generations=ggaparameters.Ge,
            num_parents_mating=ggaparameters.Pe,
            fitness_func=fitness_func,
            initial_population=init_pop,
            num_genes=problem.n * 2,
            init_range_low=0,
            init_range_high=1,
            keep_parents=0,
            mutation_percent_genes=ggaparameters.Pm * 100,
            save_best_solutions=True,
            gga_crossover_type="atc_wrc"
        )
        ga_mpc.gga_run()

        best_sol_genid = ga_mpc.best_solution_generation
        best_gene = ga_mpc.best_solutions[best_sol_genid]
        solution = Solution.decode(problem, best_gene)
        print(solution.f3)
        print(solution.QI)
        result3.append(solution.QI)
    print(ns)
    print(result1)
    print(result2)
    print(result3)

    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.plot(ns, result1, label='GGA (WRC)')
    plt.plot(ns, result2, label='GGA (ATC)')
    plt.plot(ns, result3, label='GGA (WRC & ATC)')
    plt.legend()
    plt.xlabel("n (Problem Size)")
    plt.ylabel("QI")
    plt.title("Problem Size vs. QI")
    plt.show()

    """
    ns = [20, 30, 40, 50, 60, 70, 80, 90, 100]
    # WRC
    r1 = [0.5720384204909285, 0.6015569709837226, 0.6253203485392107, 0.6197654941373535, 0.537037037037037, 0.4680534918276374, 0.45876946642838906, 0.46241830065359474, 0.5589198036006546]
    # SPC
    r2 = [0.5112059765208111, 0.6036801132342534, 0.5545873910814967, 0.5871021775544389, 0.4919636617749825, 0.4101040118870728, 0.4204748532039826, 0.39309056956115773, 0.5147299509001637]
    # TPC
    r3 = [0.5112059765208111, 0.5739561217268224, 0.5607380830343414, 0.5921273031825797, 0.4825296995108316, 0.4404160475482912, 0.4235384222619351, 0.4140989729225023, 0.49018003273322425]
    # ATC
    r4 = [0.5464247598719317, 0.5687679083094556, 0.5606936416184971, 0.5586497890295359, 0.45217689406924927, 0.512396694214876, 0.4394771241830065, 0.502667852378835, 0.49635332252836306]
    # ATC & WRC
    r5 = [0.5464247598719317, 0.6010028653295129, 0.5606936416184971, 0.5358649789029536, 0.5241686664381214, 0.5380755608028335, 0.4394771241830065, 0.5046687416629613, 0.49817666126418153]
    """
