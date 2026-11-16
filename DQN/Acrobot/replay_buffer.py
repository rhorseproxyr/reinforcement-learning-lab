from collections import deque
import random

class ReplayBuffer:

    def __init__(self, capacity, n_step=0, gamma=False):
        self.memory = deque(maxlen=capacity)
        
        if n_step > 0:
            self.n_step_mode = True
            self.n_step = n_step
            self.pending = deque(maxlen=n_step)
            if gamma == False:
                print("WARNING ; GAMMA FOR N_STEP MODE MUST BE ENABLE, USING 0.99 VALUE FOR GAMMA")
                self.gamma = 0.99
            else:
                self.gamma = gamma
                
                
            
    def push(self, state, action, reward, new_state, terminated):
        if not self.n_step_mode:
            self.memory.append((
                state.copy(),
                action,
                reward,
                new_state.copy(),
                terminated
            ))
        else:
            self.pending.append((
                state.copy(),
                action,
                reward,
                new_state.copy(),
                terminated
            ))
            if len(self.pending) == self.n_step:
                first = self.pending[0]
                state, action, reward, new_state ,terminated = first
                if terminated:
                    self.memory.append((
                        state.copy(),
                        action,
                        reward,
                        new_state.copy(),
                        terminated
                    ))
                else:
                    all_reward = 0
                    counting = 1
                    new_state = None
                    for N in reversed(self.pending):
                        all_reward += (self.gamma * counting) * N[2]
                        new_state = N[3]
                        counting += 1
                        if N[4]:
                            break
                    self.memory.append((
                        state.copy(),
                        action,
                        all_reward,
                        new_state.copy(),
                        terminated
                    ))
                    
                self.pending.popleft()
                                        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
