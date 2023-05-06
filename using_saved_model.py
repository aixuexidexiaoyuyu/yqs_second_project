import gym
from stable_baselines3 import DQN
# from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
# from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make("LunarLander-v2")
model = DQN.load("./model/LunarLander3.pkl")

state = env.reset()
done = False
score = 0
episode = 1
while not done:
    action, _ = model.predict(observation=state)
    state, reward, done, info = env.step(action=action)
    score += reward
    env.render()
print("Episode : {}, Score : {}".format(episode, score))
env.close()