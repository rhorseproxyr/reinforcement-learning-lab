from collections import deque
import random

class ReplayBuffer:

    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, new_state, terminated):
        self.memory.append((
            state.copy(),
            action,
            reward,
            new_state.copy(),
            terminated
        ))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)