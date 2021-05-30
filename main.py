import numpy as np
import matplotlib.pyplot as plt

from pcb import Problem, Solution
from gga import GGA
from ggaparam import GGAParameters

def fitness_func(gene: np.ndarray, _: int) -> float:
    sol = Solution.decode(problem, gene)

    # pygad maximizes the fitness function, so we add a minus here
    return -sol.f


if __name__ == "__main__":
    np.random.seed(43)

    ggaparameters = GGAParameters()
    ns = []
    r1 = []
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

        # Survey Stage
        ga_survey_instance = GGA(
            problem_ins=problem,
            solution_cls=Solution,
            ggaparameters=ggaparameters,
            num_generations=ggaparameters.Gs,
            num_parents_mating=ggaparameters.Ps,
            fitness_func=fitness_func,
            sol_per_pop=ggaparameters.Ps,
            num_genes=problem.n * 2,
            init_range_low=0,
            init_range_high=1,
            keep_parents=0, # Survey stage
            mutation_percent_genes=ggaparameters.Pm * 100,
            save_best_solutions=False,
        )
        ga_survey_instance.gga_run()

        # select best Pe from survey result
        Pe = ggaparameters.Pe
        survey_pop = ga_survey_instance.population
        survey_fit = ga_survey_instance.last_generation_fitness
        init_pop = survey_pop[survey_fit.argsort()[::-1][:Pe]]

        # Evolution Stage
        ga_instance = GGA(
            problem_ins=problem,
            solution_cls=Solution,
            ggaparameters=ggaparameters,
            num_generations=ggaparameters.Ge,
            num_parents_mating=ggaparameters.Pe,
            fitness_func=fitness_func,
            initial_population=init_pop,
            keep_parents=1,
            mutation_percent_genes=ggaparameters.Pm * 100,
            save_best_solutions=True,
        )
        ga_instance.gga_run()


        best_sol_genid = ga_instance.best_solution_generation
        best_gene = ga_instance.best_solutions[best_sol_genid]
        solution = Solution.decode(problem, best_gene)
        print(solution.f)
        print(solution.QI)
        r1.append(solution.QI)


    fig = plt.figure()
    plt.plot(ns, r1, label='GGA')
    plt.legend()
    plt.xlabel("n (Problem Size)")
    plt.ylabel("QI")
    plt.title("Problem Size vs. QI")
    plt.show()
