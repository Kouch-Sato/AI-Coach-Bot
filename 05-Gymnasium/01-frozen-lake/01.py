import gymnasium as gym

env = gym.make(
    "FrozenLake-v1",
    map_name = "4x4",
    is_slippery = False
)

obs, info = env.reset()

# 0:left, 1:down, 2:right, 3:top
action = 1

next_obs, reward, terminated, truncated, info = env.step(action)

print(next_obs)
print(reward)