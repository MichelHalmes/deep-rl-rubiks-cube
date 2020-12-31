import logging
import sys

from ..rubiks.cube_wrapper import MyCube
from ..dqn_solver import DqnCubeSolver
from ..a2c_solver import A2cCubeSolver


def main():
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s")

    cube = MyCube()
    rl_solver = DqnCubeSolver(cube)
    rl_solver.train()


if __name__ == "__main__":
    main()
