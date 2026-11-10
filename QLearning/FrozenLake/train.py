import gymnasium as gym
import numpy as np

env = gym.make(
    "FrozenLake-v1",
    map_name="4x4",
    is_slippery=True,
    render_mode="ansi"
)

state_space = env.observation_space.n
action_space = env.action_space.n

gamma = 0.9
learning_rate = 0.1

epsilon_start = 1.0
epsilon_min = 0.1

total_episodes = 10000
total_wins = 0

q_table = np.zeros((state_space, action_space))

# Training
for episode in range(total_episodes):

    # Slowing epsilon
    epsilon = epsilon_start - (
        (epsilon_start - epsilon_min) * episode / total_episodes
    )

    state, info = env.reset()
    done = False

    while not done:

        if np.random.random() < epsilon:
            # Exploration
            action = env.action_space.sample()
        else:
            # Exploitation
            action = np.argmax(q_table[state])

        new_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if done and reward == 1:
            total_wins += 1
          
        old_q_value = q_table[state, action]
      
        max_next_q_value = (
            0 if done else np.max(q_table[new_state])
        )

        # Formula QLearning
        new_q_value = old_q_value + learning_rate * (reward + gamma * max_next_q_value - old_q_value)
      
        q_table[state, action] = new_q_value
      
        state = new_state


print("Nombre de victoires :", total_wins)
print("Taux de victoire :",total_wins / total_episodes * 100,"%")

# Evaluate the agent
test_wins = 0
total_tests = 100

for test_episode in range(total_tests):
    state, info = env.reset()
    done = False

    while not done:
        action = np.argmax(q_table[state])

        new_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        state = new_state

    if reward == 1:
        test_wins += 1

print("Victoires en test :", test_wins, "/", total_tests)

# Saving q_table
np.save("q_table.npy", q_table)

env.close()
