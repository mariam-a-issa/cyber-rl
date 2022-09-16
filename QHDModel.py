from __future__ import print_function, division
import torch
import random
import copy
import numpy as np

class QHDModel(object):
    def __init__(self, n_obs, n_actions, lr=0.05, lr_decay_rate=0.95, dimension=1000):
        self.n_obs = n_obs
        self.n_actions = n_actions
        self.D = dimension
        self.lr=lr
        self.lr_decay_rate = lr_decay_rate

        # Initialize HDModel
        self.model = []
        for a in range(self.n_actions):
            self.model.append(np.zeros(self.D, dtype=complex))

        # Initialize HDVec
        self.s_hdvec = []
        for n in range(self.n_obs):
            self.s_hdvec.append(np.random.normal(0, 1, self.D))

        # Initialize Bias
        self.bias = np.random.uniform(0,2*np.pi, size=self.D)

        # Initialize Delay Model
        self.model = np.array(self.model)
        self.delay_model = torch.from_numpy(copy.deepcopy(self.model))

        # Convert to tensors
        self.s_hdvec = torch.FloatTensor(self.s_hdvec)
        self.model = torch.tensor(self.model, dtype=torch.cfloat)
        self.delay_model = torch.tensor(self.delay_model, dtype=torch.cfloat)
        self.bias = torch.FloatTensor(self.bias)

    def update_delay_model(self):
        self.delay_model = self.model.detach().clone()

    # def keep_best_model(self, best_reward, best_step, current_reward, current_step):
    #     if current_reward > best_reward:
    #         best_reward = current_reward
    #         best_step = current_step
    #     return best_reward, best_step

    def get_q_value(self, state):
        # if torch.rand(1) <= epsilon:
        #     return np.random.choice(self.n_actions)
        # else:
            # if type(state) == torch.Tensor:
            #     state = state.to(self.GPU_device)
            # else:
            #     state = torch.tensor(state, dtype=torch.float64).to(self.GPU_device)

        # Calculate Best Q Value
        encoded = torch.exp(1j* (torch.matmul(state, self.s_hdvec) + self.bias))  #(1 x n_obs)*(n_obs x D)=(1 x D)
        q_values = torch.real(torch.matmul(torch.conj(encoded), self.model.t()) / self.D)  # (1 x D)*(D x n_actions)=(1 x n_actions)
        # best_action = int(torch.argmax(q_values))
        return q_values

    def update_model(self, buffer, batch_size, attacker=True, gamma=0.99):
        minibatch = buffer.sample(batch_size)

        s_attack, s_defend, a_attack, a_defend, r_attack, r_defend, is_done, ns_attack, ns_defend = minibatch

        # Differentiate between attack and defense
        if attacker:
            rewards = torch.tensor(r_attack).float()
            states = torch.tensor(s_attack).float()
            next_states = torch.tensor(ns_attack).float()
            actions = torch.tensor(a_attack).float()
        else:
            rewards = torch.tensor(r_defend).float()
            states = torch.tensor(s_defend).float()
            next_states = torch.tensor(ns_defend).float()
            actions = torch.tensor(a_defend).float()
        is_done = torch.tensor(is_done)

        # Move to GPU if using GPU
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            rewards = rewards.to(device)
            states = states.to(device)
            next_states = next_states.to(device)

        # Encode Current and Next States
        encoded = torch.exp(1j* (torch.matmul(states, self.s_hdvec) + self.bias))
        encoded_ = torch.exp(1j* (torch.matmul(next_states, self.s_hdvec) + self.bias))

        # Predict State
        actions = [int(a) for a in actions]
        y_pred = torch.real(torch.sum(torch.conj(encoded) * self.model[actions], dim=1) / self.D)

        a_ = np.arange(self.n_actions)
        q_values = torch.real(torch.matmul(torch.conj(encoded_), self.delay_model[a_].t()) / self.D)
        max_q_values, _ = torch.max(q_values, dim=1)

        # Use Bellman Equation to calculate actual loss
        y_true = rewards + gamma * max_q_values
        y_true = torch.where(is_done, rewards, y_true)  # if done, then we use actual reward, not Bellman Eqn Estimate

        loss = torch.square(y_true - y_pred)

        for i, action in enumerate(actions):
            self.model[action] += self.lr * (y_true[i] - y_pred[i]) * encoded[i, :]

        return torch.mean(loss)
