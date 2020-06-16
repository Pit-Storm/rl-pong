# RL Pong A3C - Code to University term paper

This repository shows the work of a university term paper. The goal was to let a RL-Agent learn how to play the Atari game PONG. With the trained model(s), there had been made some variance tests to gather deeper understanding about the A3C Algorithm.

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

The results of the training process is provided for [download](). You have to unpack the archive into your local repository. The folder `results` should only contain two folder `disadvantage` and `advantage` with a few other folders inside.

## Reproduce Analysis

Just run the notebooks `02_*` and above to check the analysis.
