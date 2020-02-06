import logging
import sys

from ..rubiks.cube_wrapper import MyCube
from ..rl_solver import RlCubeSolver


def main():
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s")

    cube = MyCube()
    rl_solver = RlCubeSolver(cube)
    rl_solver.train()


if __name__ == "__main__":
    main()
    