import dataclasses


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
