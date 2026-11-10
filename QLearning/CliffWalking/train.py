"""
The formula of rewards and victory of episode is only working for CliffWalking, to understand see CliffWalking configuration (gymnasium)
"""
import gymnasium as gym
import numpy as np

# Création de l'environnement 
env = gym.make('CliffWalking-v1', is_slippery=True)

# Les variables pour l'entrainement
state_space = env.observation_space.n
action_space = env.action_space.n
gamma = 0.95
learning_rate = 0.1

epsilon_start = 1.0
epsilon_min = 0.05

total_episodes = 10000
max_step = 200

best_rewards = float("-inf")
episode_rewards = 0

q_table = np.zeros((state_space, action_space))

# Evaluate function
def evaluate(evaluate_episodes=100):
    total_rewards = 0
    total_success = 0
    for episode in range(evaluate_episodes):
        state, _ = env.reset()

        for _ in range(max_step):
            new_state, reward, terminated, truncated, _ = env.step(action=np.argmax(q_table[state]))
            state = new_state

            total_rewards+=reward
            if terminated or truncated:
                break

        if terminated:
            total_success += 1

    print(f"Réussites : {total_success}/{evaluate_episodes}")
    print(f"Récompense moyenne : {total_rewards / evaluate_episodes:.2f}")
    
# Training loop 
for episode in range(total_episodes):
    state, _ = env.reset()

    episode_rewards = 0  

  
    epsilon = epsilon_start - ((epsilon_start - epsilon_min) * episode / total_episodes) 

    # Explorate / Exploitation
    for step in range(max_step):
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])


        new_state, reward, terminated, truncated, _ = env.step(action)
        episode_rewards += reward

        old_q_value = q_table[state, action] # Ancienne valeur Q de l'action effectuée

        max_next_q_value = 0 if terminated else np.max(q_table[new_state]) # Meilleure valeur Q de l'état suivant

        new_q_value = old_q_value + learning_rate * (reward + gamma * max_next_q_value - old_q_value)

        q_table[state, action] = new_q_value

        state = new_state

        if terminated or truncated:
            break

    # Data while training
    if episode_rewards > best_rewards:
        best_rewards = episode_rewards
        print(f"👑 New best episode reward : {best_rewards}")

    if (episode + 1) % (total_episodes // 8) == 0:
        print(f"Episode {episode + 1}/{total_episodes} ; Gamma {gamma} ; Epsilon {epsilon}")
        evaluate()

np.save("q_table.npy", q_table)
print("Q-table sauvegardée !")


# Final evaluation after training
test_wins = 0

for test_episode in range(100):
    state, info = env.reset()
    episode_rewards = 0

    for step in range(max_step):

        action = np.argmax(q_table[state])

        new_state, reward, terminated, truncated, info = env.step(action)
        episode_rewards += reward
        state = new_state

        if terminated or truncated:
            break


    
    if terminated:
        test_wins += 1

print(f"Victoires en test : {test_wins}/ 100")


env.close()
