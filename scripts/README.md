# A3C Blog Post

Code related to [the Blogpost](https://blog.tensorflow.org/2018/07/deep-reinforcement-learning-keras-eager-execution.html). See the [related repo](https://github.com/tensorflow/models/tree/master/research/a3c_blogpost) for the original code. The Code in this repo here had been modified by the author to meet the needs for the term paper.

Running the script:
* You can see all paramaters with you can run the script by executing `python a3c_research.py --help`.
* Train a model with default parameters: `python a3c_research.py --train`
* Play the game with the trained network: `python a3c_research.py`
* Compare the network performance to a random agent: `python a3c_research.py --train --algorithm "random"`
