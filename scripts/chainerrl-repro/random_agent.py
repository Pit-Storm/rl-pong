import gym
from chainerrl.agent import Agent

class random_agent(Agent):
    def __init__(self, envname):
        self.env = envname
        super().__init__()

    def act(self, obs):
        return gym.make(self.env).action_space.sample()

    def act_and_train(self):
        None
    
    def get_statistics(self):
        return [
            ('random_agent', 'No internal stats'),
        ]
    
    def load(self):
        None
    
    def save(self):
        None
    
    def stop_episode(self):
        None
    
    def stop_episode_and_train(self):
        None