import numpy as np
import gymnasium as gym

class QLearningAgent():
  def __init__(self, qtable, gamma, lr, env, eps_min):
    self.qtable = qtable
    self.gamma = gamma
    self.lr = lr 
    self.env = env
    self.eps_min = eps_min
    self.eps_max = 1
    self.epsilon = 1
    
  def update_q_table(self, reward, state, new_state, action, terminated):
    max_next_q_value = 0 if terminated else np.max(self.qtable[new_state])
    target = reward + self.gamma * max_next_q_value
    error = target - self.qtable[state, action]
    self.qtable[state, action] += self.lr * error
    
  def choose_action(self, state, exploration=True):
    if exploration == True and np.random.random() < self.epsilon:
      return self.env.action_space.sample()
    else:
      return np.argmax(self.qtable[state])

  def training(self, max_step, max_episode,):
    for episode in range(max_episode):
      state, _ = self.env.reset()
      for step in range(max_step):
        action = self.choose_action(state)
        new_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        self.update_q_table(reward, state, new_state, action, terminated)
        if done:
          break
        state = new_state
        
    
    
