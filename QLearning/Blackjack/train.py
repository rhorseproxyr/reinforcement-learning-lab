import gymnasium as gym
import numpy as np

class QLearningAgent():
  def __init__(self, qtable, gamma, lr, env, eps_min):
    self.qtable = qtable
    self.gamma = gamma
    self.lr = lr 
    self.env = env
    self.eps_min = eps_min
    self.eps_max = 1
    self.epsilon = 1
    self.best_mean_rewards = float("-inf")
  def update_q_table(self, reward, state, new_state, action, terminated):
    max_next_q_value = 0 if terminated else np.max(self.qtable[new_state])
    target = reward + self.gamma * max_next_q_value
    error = target - self.qtable[state + (action,)]
    self.qtable[state +(action,)] += self.lr * error

  def calculate_epsilon(self, current_episode, max_episode):
    self.epsilon = self.eps_max - ((self.eps_max - self.eps_min) * current_episode / max_episode)
                                       
  def choose_action(self, state, exploration=True):
    if exploration == True and np.random.random() < self.epsilon:
      return self.env.action_space.sample()
    else:
      return np.argmax(self.qtable[state])
      
  def evaluate_agent(self, max_evaluation_episode, max_evaluation_step):
    total_rewards = 0
    for episode in range(max_evaluation_episode):
      state, _ = self.env.reset()
      for step in range(max_evaluation_step):
        action = self.choose_action(state, exploration=False)
        new_state, reward, terminated, truncated, _ = self.env.step(action)
        total_rewards += reward
        state = new_state
        if terminated or truncated:
          break
    mean_rewards = total_rewards / max_evaluation_episode
    if mean_rewards > self.best_mean_rewards:
      self.best_mean_rewards = mean_rewards
      print(f"👑 New best agent detected with {mean_rewards:.2f} mean reward")
      np.save("q_table.npy", self.qtable)
        
  def training(self, max_step, max_episode):
    print("Training...")
    for episode in range(max_episode):
      state, _ = self.env.reset()
      self.calculate_epsilon(episode, max_episode)
      for step in range(max_step):
        action = self.choose_action(state)
        new_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        self.update_q_table(reward, state, new_state, action, terminated)
        if done:
          break
        state = new_state

      if (episode + 1) % (max_episode // 50) == 0:
        print(f"Episode {episode+1}/{max_episode}\nEpsilon:{self.epsilon}")
        self.evaluate_agent(max_evaluation_episode=10000, max_evaluation_step=500)

env = gym.make("Blackjack-v1")

state_space = [space.n for space in env.observation_space.spaces] 
action_space = env.action_space.n 

q_table = np.zeros((*state_space, action_space))

agent = QLearningAgent(qtable=q_table,gamma=0.95,lr=0.1,env=env,eps_min=0.05)

agent.training(max_step=500,max_episode=150000)
