import torch
import torch.optim as optim
import random
import numpy as np

from model import DQN
from replay_buffer import ReplayBuffer

class DQNAgent:
    def __init__(self, gamma, lr, epsilon, capacity, env, n_observations, n_actions):
        # Utiliser le gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("DQN utilise :", self.device)

        self.gamma = gamma
        self.lr = lr 
        self.epsilon = epsilon
        self.env = env
        # Réseaux nn
        self.online_network = DQN(n_input=n_observations,n_output=n_actions).to(self.device)
        self.target_network = DQN(n_input=n_observations,n_output=n_actions).to(self.device)
        self.target_network.load_state_dict(self.online_network.state_dict()) # Initialiser le target avec les poids du online network

        # Memoire 
        self.memory = ReplayBuffer(capacity)

        # Optimizer 
        self.optimizer = optim.Adam(self.online_network.parameters(), lr=self.lr)

    def choose_action(self, state, exploration=True):
        if exploration and random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)

            with torch.no_grad(): #  Désactive le suivi des grandients pour éviter a PyTorch de suivre un graphe des calculs
                q_values = self.online_network(state_tensor)

            return torch.argmax(q_values).item()
        
    def learn(self, batch_size):
        if len(self.memory) < batch_size:
            return False

        # Récupérér un batch d'experience dans la mémoire
        experiences = self.memory.sample(batch_size)

        # Décomposer la liste en plusieurs tuples et les regroupe avec zip
        states, actions, rewards, new_states, terminated = zip(*experiences)

        # Conversion des groupes en tenseurs
        states_tensor = torch.tensor(np.array(states), dtype=torch.float32,device=self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.int64,device=self.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32,device=self.device)
        new_states_tensor = torch.tensor(np.array(new_states), dtype=torch.float32,device=self.device)
        terminated_tensor = torch.tensor(terminated, dtype=torch.bool,device=self.device)

        # Prédire les valeurs Q de toute les actions
        q_values = self.online_network(states_tensor)

        # Récupérer les valeurs Q des actions réellement affectué 
        current_q_values = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

        # Récupérer les valeurs Q cibles
        with torch.no_grad():
            next_q_values = self.target_network(new_states_tensor)
            max_next_q_values = next_q_values.max(dim=1).values

        # Permet d'anuler les valeurs futurs des états terminaux 
        mask = (~terminated_tensor).float()

        # Calculer les targets
        target_q_values = rewards_tensor + self.gamma * max_next_q_values * mask

        # Calcul de la perte
        loss = torch.nn.functional.mse_loss(current_q_values, target_q_values)

        # Réinitialiser les gradients de l'optimiseur
        self.optimizer.zero_grad() # Remettre les gradiants à zéro

        # Calculer les nouveaux gradients avec backward()
        loss.backward() # Calculer les gradients

        torch.nn.utils.clip_grad_norm_(
            self.online_network.parameters(),
            max_norm=10.0
        )

        # Mettre à jour les poids avec optimizer.step()
        self.optimizer.step()    

        return True        

    def update_target_network(self):
        self.target_network.load_state_dict(self.online_network.state_dict())

    def save(self, filename="dqn_model.pth"):
        torch.save(self.online_network.state_dict(), filename)

    def load(self, filename="dqn_model.pth"):
        self.online_network.load_state_dict(torch.load(filename, weights_only=True))
        self.update_target_network()

