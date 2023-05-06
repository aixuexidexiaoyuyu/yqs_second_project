import gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
# from stable_baselines3.common.evaluation import evaluate_policy


env = gym.make("LunarLander-v2")
env = DummyVecEnv([lambda: env])  #对env进行包装，完成环境的向量化
model = DQN(
    "MlpPolicy",
    env=env,
    learning_rate=5e-4,
    batch_size=128,
    buffer_size=50000,
    learning_starts=0,
    target_update_interval=250,
    policy_kwargs={"net_arch" : [256, 256]},
    verbose=1,
    tensorboard_log="./tensorboard/LunarLander-v2/"
)

model.learn(total_timesteps=1e5) #总共的时间步的数量

model.save("./model/LunarLander3.pkl")

print("test for second time")


