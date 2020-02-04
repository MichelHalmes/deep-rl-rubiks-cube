import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from cube_wrapper import MyCube

EPS = .05
EPS_1 = 1. + EPS
LABEL_COLOR = "#7f00ff"
PLASTIC_COLOR = "#1f1f1f"
COLOR_DICT = {
    "W": "#ffffff",
    "Y": "#ffcf00",
    "B": "#00008f",
    "G": "#009f0f",
    "O": "#ff6f00",
    "R": "#cf0000"
}

def plot_cube(cube):
    fig, ax = _init_fig(cube.SIZE)
    _draw_sides(cube, ax)
    fig.savefig("data/cube.png", dpi=865 / cube.SIZE)


def _init_fig(cube_size):
    xlim = (-EPS_1-EPS, 3*EPS_1) 
    ylim = (-EPS_1-EPS, 2*EPS_1)
    figsize = ((xlim[1] - xlim[0]) * cube_size / 5., (ylim[1] - ylim[0]) * cube_size / 5.)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes((0, 0, 1, 1), frameon=False, xticks=[], yticks=[])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return fig, ax


def _iter_sides_offsets_name(sides):
    yield sides.right, (EPS_1, 0.),   "R"
    yield sides.left,  (-1-EPS, 0.),  "L"
    yield sides.up,    (0., EPS_1),   "U"
    yield sides.down,  (0., -1-EPS),  "D"
    yield sides.front, (0., 0.),      "F"
    yield sides.back,  (2*EPS_1, 0.), "B"


def _draw_sides(cube, ax):
    sides = cube.get_state()
    for side, offsets, name in _iter_sides_offsets_name(sides):
        cubie_size = 1. / cube.SIZE
        X0, Y0 = offsets
        for x_idx in range(cube.SIZE):
            for y_idx in range(cube.SIZE):
                color = COLOR_DICT[side[x_idx, y_idx]]
                ax.add_artist(Rectangle((X0 + x_idx*cubie_size, Y0 + y_idx*cubie_size), 
                                cubie_size, cubie_size, ec=PLASTIC_COLOR, fc=color))
        ax.text(X0 + 0.5, Y0 + 0.5, name , color=LABEL_COLOR, ha="center", va="center", fontsize=10)


if __name__ == "__main__":
    cube = MyCube()
    cube.shuffle()
    plot_cube(cube)