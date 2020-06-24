# RL Pong A3C - Code to University term paper

This repository shows the work of a university term paper. The goal was to let a RL-Agent learn how to play the Atari game PONG. With the trained model(s), there had been made some variance tests to gather deeper understanding about the A3C Algorithm and the advantage function.

## Setup

1. Python 3.7x
2. Create a virtualenvironment
3. Install the pip-packages
   * On Linux its easy: `pip install -r requirements-linux.txt`
      * If you are getting an error, when running the script, that there are not allowed operations. Your CPU doen't support the machine operations which are tensorflow (TF>=v1.4) had been compiled for. For such older CPUs, with non AVX-Instructionset support, one can install a community build. E.g. [this one for tf 1.14.1 for linux x86/x64](https://github.com/yaroslavvb/tensorflow-community-wheels/issues/132)
   * On Windows you have to do some steps before you can install all requirements:
     1. Install atari-py on windows with the guide in [this answer on github issue](https://github.com/openai/gym/issues/1726#issuecomment-550580367)
     2. After that you can run `pip install -r requirements.txt` and hopefully all worked.

## Run the code

Run `jupyter notebook` in the virtualenv to check the example and discovery notebooks.

The script in the dir `scripts` has to be run in the terminal. Check the Readme there for further reading.

## Get the results

The results of the training process is provided for [download](). You have to unpack the archive into your local repository root. The folders `results` und `models` than should only contain three folders `disadvantage`, `advantage` and `random` with a few other folders/files inside.

## Reproduce Analysis

Just run the notebook `notebooks/02_evaluation.ipynb` to check the analysis.

## Reproduce results

Just run the training from repo root with the commands you get from my results folder.

## Run a demo

* For A3C-Agent: `python scripts/chainerrl-repro/train_a3c.py --demo --render --env PongNoFrameskip-v4 --outdir results/advantage --load models/advantage`
* For advanced REINFORCE-Agent: `python scripts/chainerrl-repro/train_a3c.py --demo --render --env PongNoFrameskip-v4 --outdir results/disadvantage --load models/disadvantage`