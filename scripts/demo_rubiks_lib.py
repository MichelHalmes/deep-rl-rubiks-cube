
import sys
from os import path
import random
import time

from src.rubiks.lib import solve
from src.rubiks.lib.cube import Cube
from src.rubiks.lib.solve import Solver
from src.rubiks.lib.optimize import optimize_moves

SOLVED_CUBE_STR = "OOOOOOOOOYYYWWWGGGBBBYYYWWWGGGBBBYYYWWWGGGBBBRRRRRRRRR"
MOVES = ["L", "R", "U", "D", "F", "B", "M", "E", "S"]

# Copied from: https://github.com/pglass/cube/blob/master/solve_random_cubes.py

def random_cube():
    """
    :return: A new scrambled Cube
    """
    scramble_moves = " ".join(random.choices(MOVES, k=200))
    a = Cube(SOLVED_CUBE_STR)
    a.sequence(scramble_moves)
    return a


def run():
    successes = 0
    failures = 0

    avg_opt_moves = 0.0
    avg_moves = 0.0
    avg_time = 0.0
    while True:
        C = random_cube()
        solver = Solver(C)

        start = time.time()
        solver.solve()
        duration = time.time() - start

        if C.is_solved():
            opt_moves = optimize_moves(solver.moves)
            successes += 1
            avg_moves = (avg_moves * (successes - 1) + len(solver.moves)) / float(successes)
            avg_time = (avg_time * (successes - 1) + duration) / float(successes)
            avg_opt_moves = (avg_opt_moves * (successes - 1) + len(opt_moves)) / float(successes)
        else:
            failures += 1
            print(f"Failed ({successes + failures}): {C.flat_str()}")

        total = successes + failures
        if total == 1 or total % 100 == 0:
            pass_percentage = 100 * successes / total
            print(f"{total}: {successes} successes ({pass_percentage:0.3f}% passing)"
                  f" avg_moves={avg_moves:0.3f} avg_opt_moves={avg_opt_moves:0.3f}"
                  f" avg_time={avg_time:0.3f}s")

def main():
    solve.DEBUG = False
    run()

if __name__ == '__main__':
    main()
    
