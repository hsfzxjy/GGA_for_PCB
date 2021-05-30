# GGA_for_PCB

Implementation of Guided Genetic Algorithm (GGA), which can efficiently solve some combination optimization problems.

## Getting Started
### Prerequisites

Python 3.6 or higher must be installed.

Install the dependencies:

```bash
pip install -r requirements.txt
```
### Run

```bash
python main.py
```

After a successful run, the results will be printed in the console.

### Algorithm Paremeters

```py
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
    Ge: int = 160

    # will be selected based on the experiments, [0,1], 0.3
    alpha: float = 0.4
```

See `ggaparam.py` for more information.
## Algorithm Overview

There are mainly two stages in GGA: the survey stage and the evolution stage.

In both of the two stages, GGA employs roulette wheel selection as the parents selection method, ATC (Asexual Transposition Crossover), WRC (Weighted Recombination Crossover) as the crossover methods, Inversion Mutation, Shuffle Mutation as the mutation methods.

In the survey stage, GGA runs with a larger population size for a short time.
In the evolution stage, GGA runs roughly the same as classic GA. The difference is that the alpha-survive mechanism is leveraged.

See `gga.py` for more information.
## Problem and solution defination

We implemented the PCB assembly problem stated in [1] as an example.

Most of the procedures were vectorized by numpy, and some others (e.g. `total_sw()`, which counts the times of switches on all machines with KTNS policy) were optimized by Cython for higher performance.

For a new optimization problem to be solved by GGA, the following components should be overwritten:

- `Solution`: the class of the solution, including encoding/decoding method and coding scheme.
- `Problem`: the class of the problem.
- `fitness_fun`: the objective function to be minimized by GGA.

See `main.py`, `pcb.py` and `_C_GGA.pyx` for more information.

## Referrences

Most of the code follows the algorithm described in the following papers:

[1] Van Hop N, Tabucanon M T. Improvement of Search Process in Genetic Algorithms: An Application of PCB Assembly Sequencing Problem[M]//New Optimization Techniques in Engineering. Springer, Berlin, Heidelberg, 2004: 385-409.

[2] Tang C S, Denardo E V. Models arising from a flexible manufacturing machine, part I: minimization of the number of tool switches[J]. Operations research, 1988, 36(5): 767-777.
