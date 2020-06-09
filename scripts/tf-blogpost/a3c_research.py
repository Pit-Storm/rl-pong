import warnings  
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)

    import os
    import sys

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    from tensorflow.python.util import deprecation
    deprecation._PRINT_DEPRECATION_WARNINGS = False

    import argparse
    import multiprocessing
    import threading
    from queue import Queue

    import gym
    import matplotlib.pyplot as plt
    import numpy as np

    import tensorflow as tf
    from tensorflow.python import keras
    from tensorflow.python.keras import layers

    from datetime import datetime

    import shelve

tf.enable_eager_execution()

format_str = "%Y-%m-%d_%H-%M-%S"
timestamp = datetime.now().strftime(format_str)

parser = argparse.ArgumentParser(
    description='Run A3C algorithm on game in OpenAI Gym env.')
parser.add_argument('--algorithm', default='a3c', type=str,
                    help='Choose between \'a3c\' and \'random\'.')
parser.add_argument('--train', dest='train', action='store_true',
                    help='Train our model.')
parser.add_argument('--lr', default=0.001, type=float,
                    help='Learning rate for the shared optimizer.')
parser.add_argument('--update-freq', default=20, type=int,
                    help='How often to update the global model.')
parser.add_argument('--max-eps', default=1000, type=int,
                    help='Global maximum number of episodes to run.')
parser.add_argument('--gamma', default=0.99, type=float,
                    help='Discount factor of rewards.')
parser.add_argument('--save-dir', default=f'./data/{timestamp}/', type=str,
                    help='Directory in which you desire to save the model.')
parser.add_argument('--game', default='CartPole-v0', type=str,
                    help='Choose the env in which to train the agent. Default is "CartPole-v0"')
parser.add_argument('--workers', default=multiprocessing.cpu_count(), type=int,
                    help='Set number of workers. Defaults to number of CPUs.')
args = parser.parse_args()

def plotting(moving_average_rewards, save_dir):

    plt.plot(moving_average_rewards)
    plt.ylabel('Moving average ep reward')
    plt.xlabel('Episodes')
    plt.savefig(os.path.join(save_dir,"results_plot.png"))
    # plt.show()

def save_rewards(moving_average_rewards):
    with shelve.open('./data/moving_average_rewards_models') as file:
        file[timestamp] = moving_average_rewards

class ActorCriticModel(keras.Model):
    def __init__(self, state_size, action_size):
        super(ActorCriticModel, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.dense1 = layers.Dense(1000, activation='relu')
        self.policy_logits = layers.Dense(action_size)
        self.dense2 = layers.Dense(1000, activation='relu')
        self.values = layers.Dense(1)

    def call(self, inputs):
        # Forward pass
        x = self.dense1(inputs)
        logits = self.policy_logits(x)
        v1 = self.dense2(inputs)
        values = self.values(v1)
        return logits, values


def record(episode,
           episode_reward,
           worker_idx,
           global_ep_reward,
           result_queue,
           total_loss,
           num_steps):
    """Helper function to store score and print statistics.

    Arguments:
      episode: Current episode
      episode_reward: Reward accumulated over the current episode
      worker_idx: Which thread (worker)
      global_ep_reward: The moving average of the global reward
      result_queue: Queue storing the moving average of the scores
      
      total_loss: The total loss accumulated over the current episode
      num_steps: The number of steps the episode took to complete
    """
    if global_ep_reward == 0:
        global_ep_reward = episode_reward
    else:
        global_ep_reward = global_ep_reward * 0.99 + episode_reward * 0.01
    print(
        f"Episode: {episode} | "
        f"Moving Average Reward: {int(global_ep_reward)} | "
        f"Episode Reward: {int(episode_reward)} | "
        f"Loss: {int(total_loss / float(num_steps) * 1000) / 1000} | "
        f"Steps: {num_steps} | "
        f"Worker: {worker_idx}"
    )
    
    result_queue.put(global_ep_reward)
    return global_ep_reward


class RandomAgent:
    """Random Agent that will play the specified game

      Arguments:
        env_name: Name of the environment to be played
        max_eps: Maximum number of episodes to run agent for.
    """

    def __init__(self, env_name, max_eps):
        self.env = gym.make(env_name)
        self.max_episodes = max_eps
        self.global_moving_average_reward = 0
        self.res_queue = Queue()

    def run(self):
        reward_avg = 0
        for episode in range(self.max_episodes):
            done = False
            self.env.reset()
            reward_sum = 0.0
            steps = 0
            while not done:
                # Sample randomly from the action space and step
                _, reward, done, _ = self.env.step(
                    self.env.action_space.sample())
                steps += 1
                reward_sum += reward
            # Record statistics
            self.global_moving_average_reward = record(episode,
                                                       reward_sum,
                                                       0,
                                                       self.global_moving_average_reward,
                                                       self.res_queue, 0, steps)

            reward_avg += reward_sum
        final_avg = reward_avg / float(self.max_episodes)
        print("Average score across {} episodes: {}".format(
            self.max_episodes, final_avg))
        return final_avg


class MasterAgent():
    def __init__(self):
        self.game_name = args.game
        if not args.train and not os.path.isdir(args.save_dir):
            # set save_dir to first subfolder in data dir to load a random model
            self.save_dir = os.path.join('./data/', next(os.scandir("./data")).name)
        else:
            self.save_dir = args.save_dir

        env = gym.make(self.game_name)
        self.state_size = [np.prod(env.observation_space.shape)]
        self.action_size = env.action_space.n
        self.opt = tf.train.RMSPropOptimizer(args.lr, use_locking=True)
        print("State size is - {} - and action size is - {} -".format(
            self.state_size, self.action_size))

        self.global_model = ActorCriticModel(
            self.state_size, self.action_size)  # global network
        self.global_model(tf.convert_to_tensor(
            np.random.random((1, *self.state_size)), dtype=tf.float32))

    def train(self):
        if args.algorithm == 'random':
            random_agent = RandomAgent(self.game_name, args.max_eps)
            random_agent.run()
            return

        res_queue = Queue()

        if args.workers >= multiprocessing.cpu_count():
            worker_count = multiprocessing.cpu_count()
        else:
            worker_count = args.workers

        workers = [Worker(self.state_size,
                          self.action_size,
                          self.global_model,
                          self.opt, res_queue,
                          i, game_name=self.game_name,
                          save_dir=self.save_dir) for i in range(worker_count)]

        for i, worker in enumerate(workers):
            print("Starting worker {}".format(i))
            worker.start()

        moving_average_rewards = []  # record episode reward to plot
        while True:
            reward = res_queue.get()
            if reward is not None:
                moving_average_rewards.append(reward)
            else:
                break
        [w.join() for w in workers]

        plotting(moving_average_rewards,self.save_dir)
        save_rewards(moving_average_rewards)

    def play(self):
        env = gym.make(self.game_name)
        state = env.reset()
        model = self.global_model
        model_path = os.path.join(
            self.save_dir, f'model_{self.game_name}.h5')
        print('Loading model from: {}'.format(model_path))
        model.load_weights(model_path)
        done = False
        step_counter = 0
        reward_sum = 0

        try:
            while not done:
                env.render(mode='rgb_array')
                logits, value = model(
                    tf.reshape(
                        tf.convert_to_tensor(
                            state[None, :], dtype=tf.float32),
                        [1, np.prod(np.array(state).shape)]
                    )
                )
                policy = tf.nn.softmax(logits)
                action = np.argmax(policy)
                state, reward, done, _ = env.step(action)
                reward_sum += reward
                print("{}. Reward: {}, action: {}".format(
                    step_counter, reward_sum, action))
                step_counter += 1
        except KeyboardInterrupt:
            print("Received Keyboard Interrupt. Shutting down.")
        finally:
            env.close()


class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def store(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []


class Worker(threading.Thread):
    # Set up global variables across different threads
    global_episode = 0
    # Moving average reward
    global_moving_average_reward = 0
    best_score = 0
    save_lock = threading.Lock()

    def __init__(self,
                 state_size,
                 action_size,
                 global_model,
                 opt,
                 result_queue,
                 idx,
                 game_name=args.game,
                 save_dir=args.save_dir):
        super(Worker, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.result_queue = result_queue
        self.global_model = global_model
        self.opt = opt
        self.local_model = ActorCriticModel(self.state_size, self.action_size)
        self.worker_idx = idx
        self.game_name = game_name
        self.env = gym.make(self.game_name)
        self.save_dir = save_dir
        self.ep_loss = 0.0

    def run(self):
        # total_step = 1
        mem = Memory()
        while Worker.global_episode < args.max_eps:
            current_state = self.env.reset()
            mem.clear()
            ep_reward = 0.
            ep_steps = 0
            self.ep_loss = 0

            time_count = 0
            done = False
            while not done:
                logits, _ = self.local_model(
                    tf.reshape(
                        tf.convert_to_tensor(
                            current_state[None, :], dtype=tf.float32),
                        [1, np.prod(np.array(current_state).shape)]
                    )
                )

                probs = tf.nn.softmax(logits)

                action = np.random.choice(self.action_size, p=probs.numpy()[0])
                new_state, reward, done, _ = self.env.step(action)
                if done:
                    reward = -1
                ep_reward += reward
                mem.store(np.reshape(current_state, np.prod(
                    np.array(current_state).shape)), action, reward)

                if time_count == args.update_freq or done:
                    # Calculate gradient wrt to local model. We do so by tracking the
                    # variables involved in computing the loss by using tf.GradientTape
                    with tf.GradientTape() as tape:
                        total_loss = self.compute_loss(done,
                                                       new_state,
                                                       mem,
                                                       args.gamma)
                    self.ep_loss += total_loss
                    # Calculate local gradients
                    grads = tape.gradient(
                        total_loss, self.local_model.trainable_weights)
                    # Push local gradients to global model
                    self.opt.apply_gradients(zip(grads,
                                                 self.global_model.trainable_weights))
                    # Update local model with new weights
                    self.local_model.set_weights(
                        self.global_model.get_weights())

                    mem.clear()
                    time_count = 0

                    if done:  # done and print information
                        Worker.global_moving_average_reward = \
                            record(Worker.global_episode, ep_reward, self.worker_idx,
                                   Worker.global_moving_average_reward, self.result_queue,
                                   self.ep_loss, ep_steps)
                        # We must use a lock to save our model and to print to prevent data races.
                        if ep_reward > Worker.best_score:
                            with Worker.save_lock:
                                print("Saving best model to {}, "
                                      "episode score: {}".format(self.save_dir, ep_reward))
                                self.global_model.save_weights(
                                    os.path.join(self.save_dir,
                                                 'model_{}.h5'.format(self.game_name))
                                )
                                Worker.best_score = ep_reward
                        Worker.global_episode += 1
                ep_steps += 1

                time_count += 1
                current_state = new_state
                # total_step += 1
        self.result_queue.put(None)

    def compute_loss(self,
                     done,
                     new_state,
                     memory,
                     gamma=args.gamma):
        if done:
            reward_sum = 0.  # terminal
        else:
            reward_sum = self.local_model(tf.reshape(
                tf.convert_to_tensor(
                    new_state[None, :], dtype=tf.float32),
                [1, np.prod(np.array(new_state).shape)]
            ))[-1].numpy()[0]

            # print("reward_sum: {}".format(reward_sum))

        # Get discounted rewards
        # TODO: Checkin paper what the discounted rawards are.
        discounted_rewards = []
        for reward in memory.rewards[::-1]:  # reverse buffer r
            reward_sum = reward + gamma * reward_sum
            discounted_rewards.append(reward_sum)
        discounted_rewards.reverse()

        logits, values = self.local_model(
            tf.convert_to_tensor(np.vstack(memory.states),
                                 dtype=tf.float32))

        # print("values: {}".format(values))

        # TODO: Checkin paper how the advantage must be calculated
        # TODO: Modify the advantage to a baseline when parameter has been set.
        # Get our advantages
        advantage = tf.convert_to_tensor(np.array(discounted_rewards)[:, None],
                                         dtype=tf.float32) - values
        # Value loss
        value_loss = advantage ** 2

        # Calculate our policy loss
        policy = tf.nn.softmax(logits)
        entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=policy, logits=logits)

        policy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=memory.actions,
                                                                     logits=logits)
        policy_loss *= tf.stop_gradient(advantage)
        policy_loss -= 0.01 * entropy
        total_loss = tf.reduce_mean((0.5 * value_loss + policy_loss))
        return total_loss


if __name__ == '__main__':

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    # write argv to file
    with open(os.path.join(args.save_dir, 'run_args.txt'), 'w') as f:
        f.write('\n'.join(sys.argv[1:]))

    # print(args)
    master = MasterAgent()
    if args.train:
        master.train()
    else:
        master.play()
    
    parser.exit(status=0)