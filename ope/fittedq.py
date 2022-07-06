import os

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm


class FittedQ():
    def __init__(self, sars_data, state_dim, action_dim, critic_lr, weight_decay, tau, gamma=0.99):

        self.sars_data = sars_data
        self.critic = CriticNet(state_dim+action_dim, 1)
        self.critic_target = CriticNet(state_dim+action_dim, 1)

        self.tau = tau
        self.optimizer = torch.optim.AdamW(self.critic.critic.parameters(), lr=critic_lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, "min")
        # self.optimizer = torch.optim.SGD(self.critic.parameters(), lr=critic_lr)
        self.loss = nn.MSELoss()
        self.soft_update(self.critic, self.critic_target, 1.0)


    def update(self, states, actions, next_states, next_actions, rewards, masks=None, gamma=0.99):
        next_q_inp = torch.cat((next_states, next_actions.reshape(-1, 1)), 1)
        next_q = self.critic_target(next_q_inp) / (1-gamma)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        target_q = rewards.reshape(-1, 1) + gamma * next_q

        q_inp = torch.cat((states, actions.reshape(-1, 1)), 1)
        q = self.critic(q_inp) / (1-gamma)
        critic_loss = torch.sum(torch.square(target_q - q))
        # critic_loss = self.loss(target_q, q)
        critic_loss.backward()
        self.optimizer.step()
        self.scheduler.step(critic_loss)
        self.soft_update(self.critic, self.critic_target, tau=self.tau)

        return critic_loss.item()
    
    def soft_update(self, net, target_net, tau=0.005):
        for target_param, local_param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
    
    def train_fitted_q(self, eval_policy, score="mean", train_data=None,
                       epochs=1000, log_loss=None, log_freq=10, p_bar=False):
        if train_data is None:
            train_data = self.sars_data
        train_data = train_data.to_dict("records")
        state_vars = ["outdoor_temp", "solar_irradiation", "time_hour",
                      "zone_humidity", "zone_temp", "zone_occupancy"]
        losses = []
        if p_bar:
            epoch_iterable = tqdm(range(epochs))
        else:
            epoch_iterable = range(epochs)
        for epoch in epoch_iterable:
            states = torch.zeros((len(train_data), len(state_vars)))
            next_states = torch.zeros((len(train_data), len(state_vars)))
            actions = torch.zeros((len(train_data)))  # From behavior policy
            rewards = torch.zeros((len(train_data)))
            for i, row in enumerate(train_data):
                state = [row[var] for var in state_vars]
                states[i] = torch.Tensor(state)
                rewards[i] = row["reward"]
                action = row["action"]
                next_state = [row[var+"_tp1"] for var in state_vars]

                actions[i] = action
                next_states[i] = torch.Tensor(next_state)


            next_actions = torch.Tensor(eval_policy(next_states)).sigmoid()
            loss = self.update(states, actions, next_states, next_actions, rewards)
            losses.append(loss)
            if log_loss:
                if epoch % log_freq == 0:
                    for loss in losses:
                        with open(log_loss, "a+") as f:
                            f.write(f"{loss}\n")
                    losses.clear()

        if log_loss and len(losses):
            for loss in losses:
                with open(log_loss, "a+") as f:
                    f.write(f"{loss}\n")
    
    def estimate_returns(self, eval_policy, test_data):
        state_vars = ["outdoor_temp", "solar_irradiation", "time_hour",
                      "zone_humidity", "zone_temp", "zone_occupancy"]
        states = torch.zeros((len(test_data), len(state_vars)))
        test_data = test_data.to_dict("records")
        for i, row in enumerate(test_data):
            state = [row[var] for var in state_vars]
            states[i] = torch.Tensor(state)
        actions = torch.Tensor(eval_policy(states)).sigmoid()
        inp = torch.cat((states, actions.reshape(-1, 1)), 1)
        pred_values = self.critic(inp)
        return torch.sum(pred_values)
    
    def save_params(self, log_dir):
        torch.save(self.critic.state_dict(), os.path.join(log_dir, "critic.pth"))
        torch.save(self.critic_target.state_dict(), os.path.join(log_dir, "critic_target.pth"))
    
    def load_params(self, log_dir):
        self.critic.load_state_dict(torch.load(os.path.join(log_dir, "critic.pth"), map_location=lambda storage, loc: storage))
        self.critic_target.load_state_dict(torch.load(os.path.join(log_dir, "critic_target.pth"), map_location=lambda storage, loc: storage))


class CriticNet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(CriticNet, self).__init__()

        # self.critic = nn.Sequential(
        #     nn.Linear(in_dim, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, out_dim)
        # )

        self.critic = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, out_dim)
        )
        # self.critic.apply(self.init_weights)

    def forward(self, x):
        return self.critic(x)
    
    def init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
