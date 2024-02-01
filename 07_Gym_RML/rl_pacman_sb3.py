import os.path
import gymnasium as gym
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

GAME = "ALE/Pacman-v5"
MODEL_TYPE = "DQN"  # Chose model type: PPO, DQN or A2C
TIMESTEPS = int(100_000)  # Adjust model learning time
MODEL_NAME = MODEL_TYPE + "_pacman_" + str(TIMESTEPS)


def check_file(filename):
    return os.path.isfile("./" + filename + ".zip")


def generate_model_dqn(env):
    # https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html#parameters
    return DQN("CnnPolicy", env, buffer_size=int(3e4), verbose=1)


def generate_model_ppo(env):
    # https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html#parameters
    return PPO("CnnPolicy", env, verbose=1)


def generate_model_a2c(env):
    # https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html#parameters
    return A2C("CnnPolicy", env, verbose=1)


def generate_model(game, model_type, timesteps):
    # env = gym.make(game, render_mode="rgb_array")
    # env = AtariWrapper(env)

    env = make_atari_env(game, n_envs=1, seed=0)
    env = VecFrameStack(env, n_stack=1)

    if model_type == "DQN":
        model = generate_model_dqn(env)
    elif model_type == "PPO":
        model = generate_model_ppo(env)
    else:
        model = generate_model_a2c(env)
    model.learn(total_timesteps=timesteps, progress_bar=True)
    model.save(model_type + "_pacman_" + str(timesteps))


def load_model(game, model_name, model_type):
    env = gym.make(game, render_mode="human")
    env = AtariWrapper(env)

    if model_type == "DQN":
        model = DQN.load(model_name, env=env)
    elif model_type == "PPO":
        model = PPO.load(model_name, env=env)
    else:
        model = A2C.load(model_name, env=env)
    return model


def evaluate_model(model):
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=1)
    print(mean_reward, std_reward)

    vec_env = model.get_env()
    obs = vec_env.reset()

    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render()


def main():
    if not check_file(MODEL_NAME):
        generate_model(GAME, MODEL_TYPE, TIMESTEPS)

    evaluate_model(load_model(GAME, MODEL_NAME, MODEL_TYPE))


main()