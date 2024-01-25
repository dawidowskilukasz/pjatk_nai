import gymnasium as gym
import random
import numpy as np

env = gym.make("ALE/Pacman-v5", render_mode="human")

# Get the state and action space sizes
height, width, channels = env.observation_space.shape
actions = env.action_space.n

episodes = 100
for episode in range(1, episodes + 1):
    state = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        action = env.action_space.sample()
        n_state, reward, done, truncated, info = env.step(action)
        score += reward
        print('Episode:{} Score:{}'.format(episode, score))
    env.close()
