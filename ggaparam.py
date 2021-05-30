import dataclasses


@dataclasses.dataclass
class GGAParameters:
    # Survey population size
    Ps: int = 50
    # Evolution population size
    Pe: int = 20
    # Probability of mutation
    Pm: float = .04
    # Probability of crossover
    Pc: float = .8
    # Number of survey generations
    Gs: int = 20
    # Number of evolution generations
    # Stopping condition: when the number of iterations reaches Ge
    # Ge = 50
    Ge: int = 160

    # will be selected based on the experiments, [0,1], 0.3
    alpha: float = 0.4
