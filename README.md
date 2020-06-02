# RL Pong - Code to University term paper

This repository shows the work of a university term paper. The goal is to let a RL-Agent learn how to play the Atari game PONG.

## Setup

1. Python 3.7x
2. Create two virtualenvs: one for tf2 one for tf1
3. in virtualenv tf2: `pip install -r requirements.txt` and for tf1 `pip install -r requirements-tf1.txt`
4. For older CPUs with non AVX-Instructionset support (It is needed for TF>v1.4) one can install a community build. E.g. [this one for tf 1.14.1 for linux x86/x64](https://github.com/yaroslavvb/tensorflow-community-wheels/issues/132)
5. 
## Run the code

Run `jupyter notebook` in the prefered env. Some code must be run from the terminal to work.