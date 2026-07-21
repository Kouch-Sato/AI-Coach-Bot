import gymnasium as gym
import numpy as np

env = gym.make(
    "FrozenLake-v1",
    map_name="4x4",
    is_slippery=True,
)

# 状態数: 16
state_size = env.observation_space.n

# 行動数: 4
action_size = env.action_space.n

# Q-tableをすべて0で初期化
q_table = np.zeros((state_size, action_size))

# 学習設定
episodes = 10000
learning_rate = 0.1
discount_rate = 0.99

# 最初は探索を多くする
epsilon = 1.0

# 徐々に探索を減らす
epsilon_decay = 0.999
min_epsilon = 0.01

rng = np.random.default_rng(seed=42)

for episode in range(episodes):
    state, info = env.reset()

    terminated = False
    truncated = False

    while not (terminated or truncated):
        # ε-greedy
        if rng.random() < epsilon:
            # 探索: ランダムに行動
            action = env.action_space.sample()
        else:
            # 活用: Q値が最も高い行動
            action = int(np.argmax(q_table[state]))

        next_state, reward, terminated, truncated, info = env.step(action)

        # 次の状態で最大のQ値
        best_next_q = np.max(q_table[next_state])

        # Q-learningの更新
        q_table[state, action] = q_table[state, action] + learning_rate * (
            reward
            + discount_rate * best_next_q
            - q_table[state, action]
        )

        state = next_state

    # 徐々にランダム行動を減らす
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

env.close()

np.set_printoptions(precision=2, suppress=True)
print("学習後のQ-table")
print(q_table)


symbols = {
    0: "←",
    1: "↓",
    2: "→",
    3: "↑"
}

holes = [5, 7, 11, 12]
goal = 15

for state in range(16):
    if state in holes:
        symbol = "H"
    elif state == goal:
        symbol = "G"
    else:
        best_action = np.argmax(q_table[state])
        symbol = symbols[best_action]

    print(symbol, end=" ")

    if (state + 1) % 4 == 0:
        print()


success_count = 0
test_episodes = 1000

for episode in range(test_episodes):
    state, info = env.reset()
    terminated = False
    truncated = False

    while not terminated and not truncated:
        action = np.argmax(q_table[state])

        state, reward, terminated, truncated, info = env.step(action)

    if reward == 1:
        success_count += 1

print(f"成功率: {success_count / test_episodes:.2%}")