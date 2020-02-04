import random 
import logging
from collections import namedtuple

import numpy as np

from rubik.cube import Cube, RIGHT, LEFT, UP, DOWN, FRONT, BACK 
from rubik.solve import Solver
from rubik.optimize import optimize_moves

Sides = namedtuple("Sides", ["right", "left", "up", "down", "front", "back"])
 
_SOLVED_CUBE_STR = "OOOOOOOOOYYYWWWGGGBBBYYYWWWGGGBBBYYYWWWGGGBBBRRRRRRRRR"
_MOVES = ["L", "R", "U", "D", "F", "B", "M", "E", "S", "X", "Y", "Z"]


class MyCube(object):
    """ Adapts the rubiks cube to our purposes """

    MOVES = _MOVES + [f"{m}i" for m in _MOVES]
    # FACES = [RIGHT, LEFT, UP, DOWN, FRONT, BACK]
    SIZE = 3

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

    @classmethod
    def _format_side(cls, color_list):
        return np.asarray(color_list).reshape((cls.SIZE, cls.SIZE))

    def get_state(self):
        right = [p.colors[0] for p in sorted(self._cube._face(RIGHT), key=lambda p: (-p.pos.y, -p.pos.z))]
        left  = [p.colors[0] for p in sorted(self._cube._face(LEFT),  key=lambda p: (-p.pos.y, p.pos.z))]
        up    = [p.colors[1] for p in sorted(self._cube._face(UP),    key=lambda p: (p.pos.z, p.pos.x))]
        down  = [p.colors[1] for p in sorted(self._cube._face(DOWN),  key=lambda p: (-p.pos.z, p.pos.x))]
        front = [p.colors[2] for p in sorted(self._cube._face(FRONT), key=lambda p: (-p.pos.y, p.pos.x))]
        back  = [p.colors[2] for p in sorted(self._cube._face(BACK),  key=lambda p: (-p.pos.y, -p.pos.x))]
        
        sides_fromated = [self._format_side(color_list) for color_list in (right, left, up, down, front, back)]
        return Sides(*sides_fromated)



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

