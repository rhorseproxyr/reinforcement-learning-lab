import numpy as np
import gymnasium as gym

class QLearningAgent():
  def __init__(self, qtable, gamma, lr, env, max_step, max_episode, eps_min):
    self.qtable = qtable
    self.gamma = gamma
    self.lr = lr 
    self.env = env
    self.max_step = max_step
    self.max_episode = max_episode
    self.eps_min = eps_min
    self.eps_max = 1
    self.epsilon = 1
    
  def updtate_q_table(self, reward, state, new_state, action):
    target = reward + gamma * np.max(self.qtable[new_state])
    error = target - self.qtable[state, action]
    self.qtable[state, action] += self.lr * error
    
  def choose_action(state, exploration=True):
    if exploration == True and np.random.random() < self.epsilon:
      return self.env.action_space.sample()
    else:
      return np.argmax(self.qtable[state])
    
    
