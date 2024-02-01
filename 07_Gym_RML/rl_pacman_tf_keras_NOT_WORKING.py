"""
An attempt was made to create models using the TensorFlow library and Keras API, but this ended in failure
(because of the problems resulting from the libraries versions and their compatibility). Therefore, it was decided to
use Stable Baselines3 instead.
"""

import gymnasium as gym
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Convolution2D
from keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy

env = gym.make("ALE/Pacman-v5", render_mode="human")

height, width, channels = env.observation_space.shape
actions = env.action_space.n


def build_model(height, width, channels, actions):
    model = Sequential()
    model.add(Convolution2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=(3, height, width, channels), name='Conv1'))
    model.add(Convolution2D(64, (4, 4), strides=(2, 2), activation='relu', name='Conv2'))
    model.add(Convolution2D(64, (3, 3), activation='relu', name='Conv3'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu', name='dense_512'))
    model.add(Dense(256, activation='relu', name='dense_256'))
    model.add(Dense(actions, activation='linear', name='dense_output'))
    return model


model = build_model(height, width, channels, actions)
model.summary()


def build_agent(model, actions):
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.2,
                                  nb_steps=10000)
    memory = SequentialMemory(limit=1000, window_length=3)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                   enable_dueling_network=True, dueling_type='avg',
                   nb_actions=actions, nb_steps_warmup=1000)
    return dqn


dqn = build_agent(model, actions)
dqn.compile(Adam(lr=1e-4))

dqn.fit(env, nb_steps=10000, visualize=False, verbose=2)

episodes = 100
for episode in range(1, episodes + 1):
    state = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        action = np.argmax(dqn.model.predict(np.array([state])))
        n_state, reward, done, truncated, info = env.step(action)
        score += reward
        print('Episode:{} Score:{}'.format(episode, score))

    env.close()
