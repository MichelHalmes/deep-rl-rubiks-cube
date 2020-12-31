from setuptools import setup, find_packages


version = open("src/version").read()
long_description = open("README.md").read()

install_require = [
    "matplotlib",
    "numpy",
    "Pillow",
    "torch",
]

scripts_require = []

setup_args = {
    "name": "deep_rl_rubiks_cube",
    "version": version,
    "author": "Michel Halmes",
    "author_email": "none",
    "python_requires": ">= 3",
    "description": "Solving a Rubik's-cube using Deep Reinforcement Learning and pyTorch",
    "long_description": long_description,
    "url": "https://github.com/MichelHalmes/deep-rl-rubiks-cube",
    "packages": find_packages(include=["src"]),
    "package_dir": {"src": "src"},
    "package_data": {"": ["*.md", "version"]},
    "install_requires": install_require,
    "extras_require": {"scripts": scripts_require},
    "entry_points": {
        "console_scripts": [
            "train_solver=src.entrypoints.train_solver:main",
            "demo_rubiks_lib=scripts.demo_rubiks_lib:main",
            "demo_render_cube=scripts.demo_render_cube:main",
        ],
    }
}

setup(**setup_args)