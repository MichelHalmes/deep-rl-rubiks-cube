from copy import deepcopy

from src.rubiks.cube_wrapper import MyCube
from src.rubiks.render_cube import GifRecorder

def main():
    cube = MyCube()
    cube.reset()
    cube_copy = deepcopy(cube)
    solution = cube.get_solution()
    print(cube_copy._cube)
    with GifRecorder() as recorder:
        for move in solution:
            cube_copy.step(move)
            recorder.add_frame(cube_copy)


if __name__ == "__main__":
    main()



