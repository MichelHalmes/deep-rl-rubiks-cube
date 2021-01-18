import logging
import sys
import click

from ..rubiks.cube_wrapper import MyCube
from ..dqn_solver import DqnCubeSolver
from ..a2c_solver import A2cCubeSolver
from ..ppo_solver import PpoCubeSolver

SOLVERS =  {
    "DQN": DqnCubeSolver,
    "A2C": A2cCubeSolver,
    "PPO": PpoCubeSolver,
}




@click.command(name="train_solver")
@click.option("--solver", "-S", "solver_name", default="PPO",
    type=click.Choice(SOLVERS, case_sensitive=False))
def main(solver_name):
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s")

    cube = MyCube()
    solver_cls = SOLVERS[solver_name]
    rl_solver = solver_cls(cube)
    rl_solver.train()


if __name__ == "__main__":
    main()
