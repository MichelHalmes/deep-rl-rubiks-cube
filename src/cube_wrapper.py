import random 
import logging

from rubik.cube import Cube
from rubik.solve import Solver
from rubik.optimize import optimize_moves


_SOLVED_CUBE_STR = "OOOOOOOOOYYYWWWGGGBBBYYYWWWGGGBBBYYYWWWGGGBBBRRRRRRRRR"
_MOVES = ["L", "R", "U", "D", "F", "B", "M", "E", "S", "X", "Y", "Z"]


class MyCube(object):
    """ Adapts the rubiks cube to our purposes """

    MOVES = _MOVES + [f"{m}i" for m in _MOVES]

    def __init__(self):
        self._cube = Cube(_SOLVED_CUBE_STR)

    def shuffle(self, steps=200):
        scramble_moves = " ".join(random.choices(self.MOVES, k=steps))
        self._cube.sequence(scramble_moves)
    
    def step(self, move):
        move_f = getattr(self._cube, move)
        move_f()

    def get_solution(self):
        solver = Solver(self._cube)
        solver.solve()
        assert self.is_done()
        opt_moves = optimize_moves(solver.moves)
        
        return opt_moves

    def is_done(self):
        return self._cube.is_solved()


if __name__ == "__main__":
    cube = MyCube()
    cube.shuffle()
    solution = cube.get_solution()
    print(f"Solved in {len(solution)} moves")
    assert cube.is_done()
    cube.step("L")
    cube.step("X")
    assert not cube.is_done()
    cube.step("Xi")
    cube.step("Li")
    assert cube.is_done()

