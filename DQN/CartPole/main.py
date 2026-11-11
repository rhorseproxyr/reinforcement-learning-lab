import gymnasium as gym
from agent import DQNAgent

env = gym.make("CartPole-v1")

agent = DQNAgent(
    gamma=0.99,
    lr=0.001,
    epsilon=1.0,
    capacity=10000,
    env=env
)


max_episode = 1000
max_step = 500
batch_size = 64

epsilon_min = 0.05
epsilon_decay = 0.995

target_update_frequency = 500
training_updates = 0

evaluation_frequency = 10
best_eval_reward = float("-inf")
eval_env = gym.make("CartPole-v1")

def evaluate_agent(agent, env, num_episodes=10):
    total_rewards = []

    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0

        while True:
            action = agent.choose_action(state, exploration=False)

            new_state, reward, terminated, truncated, _ = env.step(action)

            episode_reward += reward
            state = new_state

            if terminated or truncated:
                break

        total_rewards.append(episode_reward)

    return sum(total_rewards) / len(total_rewards)

# Boucle d'entrainement
for episode in range(max_episode):
    state, _ = env.reset()
    episode_reward = 0

    for step in range(max_step):
        action = agent.choose_action(state)

        new_state, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward

        # Enregistrer dans la mémoire
        agent.memory.push(state, action, reward, new_state, terminated)

        # Entrainer le nn 
        if agent.learn(batch_size):
            training_updates += 1

            if training_updates % target_update_frequency == 0:
                agent.update_target_network()

        state = new_state

        if terminated or truncated:
            break


    agent.epsilon = max(epsilon_min, agent.epsilon * epsilon_decay)

    
    print(
    f"Episode {episode + 1}/{max_episode} | "
    f"Reward: {episode_reward:.0f} | "
    f"Epsilon: {agent.epsilon:.3f}"
    )


    if (episode + 1) % evaluation_frequency == 0:
        eval_reward = evaluate_agent(agent, eval_env)

        print(f"🧪 Evaluation : {eval_reward:.1f}/500")

        if eval_reward > best_eval_reward:
            best_eval_reward = eval_reward
            agent.save("best_dqn.pth")

            print(f"👑 New best agent! Reward: {best_eval_reward:.1f}")

        
agent.save("dqn_model.pth")

agent.load("best_dqn.pth")

final_score = evaluate_agent(agent, eval_env, num_episodes=100)

print(f"🏆 Best DQN | Evaluation sur 100 épisodes : {final_score:.1f}/500")

env.close()
eval_env.close()
