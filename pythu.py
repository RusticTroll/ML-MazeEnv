import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
from gym.envs.registration import register

import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import gym.spaces as spaces

class mazeEnv(gym.Env):
    # nicknames for actions... for beginners, only
    actions = ['N','S','E','W']
    # offsets to move North, South, East, or West
    offset = [(-1,0),(1,0),(0,1),(0,-1)]

    def __init__(self):
        self.maze = np.array([[0,0,0,-1],[0,-1,0,0],[0,0,-1,0],[-1,-1,0,0]])  # the maze is hardcoded
        self.mark = None
        self.reset()
        self.observation_space = spaces.Box(np.array(1), np.array(1), dtype=np.float32)

    # clear the blocks, leaving an empty maze
    def remove_blocks(self):
        self.maze = np.zeros((4,4),dtype=int)

    # reset the environment
    def reset(self):
        self.i = 1
        self.maze = np.array([[0,0,0,-1],[0,-1,0,0],[0,0,-1,0],[-1,-1,0,0]])  # the maze is hardcoded
        self.maze[0][0] = self.i  # mark initial position with counter
        self.player = (0,0)
        return self.player

    # take one step, using a specified action
    def step(self, action):
        self.i += 1
        if type(action) is str:
            action = self.actions.index(action)
        obs = tuple(np.add(self.player, self.offset[action]))
        if max(obs) > 3 or min(obs) < 0 or self.maze[obs] == -1:
            # player is out of bounds or square is blocked... don't move player
            return self.player, -1, True
        else:
            # move was successful, advance player to new position
            self.maze[obs] = self.i
            self.player = obs
            if np.array_equal(self.player, (3,3)):  # reached the exit
                return self.player, 1, True
            else:
                return self.player, 0, False        # no outcome (player is on an open space)

    # return a random action (equally distributed across the action space)
    def sample(self):
        return self.actions[np.random.randint(4)]

    # return a random action in numeric form (equally distributed across the action space)
    def sample_n(self):
        return np.random.randint(4)

    def action_space(self):
        return spaces.Discrete(4)

    def action_space_n(self):
        return [0,1,2,3]

    def state_space(self):
        return 4,4

    def __str__(self):
        out = '\n=========='
        for x in range(4):
            out += '\n|'
            for y in range(4):
                if self.mark is not None and self.mark[0]==x and self.mark[1]==y:
                    out += '? '
                elif self.maze[x][y]>0:
                    out += str(self.maze[x][y]) + ' '
                elif self.maze[x][y]==-1:
                    out += 'X '
                elif x==3 and y==3:
                    out += '* '
                else:
                    out += '. '
            out += '|'
        out += '\n==========\n'
        return out

    def plot(q):
      fig = plt.figure(figsize=(16,6))
      ax1 = fig.add_subplot(121, projection='3d')
      x,y,b = [],[],[]
      z = np.zeros((4,4))
      for m in range(4):
        for n in range(4):
          x.append(m)
          y.append(n)
          b.append(0)
          z[m][n] += max(max(q[m][n]),0)

      ax1.bar3d(x, y, b, 1, 1, np.ravel(z), shade=True)
      plt.title('Positive Q-scores')
      plt.xlabel('state (row)')
      plt.ylabel('state (col)')
      plt.show()

print('the maze environment is all set to go')

register(
    id='mazeEnv-v1',
    entry_point='pythu:mazeEnv',
)

ENV_NAME = 'maze-v1'

# Get the environment and extract the number of actions available in the Cartpole problem
env = gym.make("mazeEnv-v1")
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

policy = EpsGreedyQPolicy()
memory = SequentialMemory(limit=50000, window_length=1)
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this slows down training quite a lot.
dqn.fit(env, nb_steps=5000, visualize=True, verbose=2)

dqn.test(env, nb_episodes=5, visualize=True)