import numpy as np
import torch
from ope.iw import InverseProbabilityWeighting
import torch.nn as nn


class GradNorm():
    def __init__(self, policy, behavior_policy):
        self.policy = policy
        self.behavior_policy = behavior_policy
    
    def calculate_loss(self, mini_batch):
        ipw = InverseProbabilityWeighting(mini_batch, retain_grad_fn=True, univariate_action=True)
        _, states, rewards, \
            policy_action_prob, behavior_action_prob = \
                ipw.evaluate_policy(self.policy.select_action, self.behavior_policy, score="")
        discounted_rewards = []
        disc_reward = 0
        for reward in rewards.tolist()[::-1]:
            disc_reward = reward + (self.policy.gamma * disc_reward)
            discounted_rewards.insert(0, disc_reward)
        states = torch.Tensor(states)
        state_values = self.policy.policy_old.critic(states)
        policy_logprob = policy_action_prob
        behavior_logprob = behavior_action_prob
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-7)
        surr1, surr2 = self.policy.calculate_loss(discounted_rewards, state_values, behavior_logprob, policy_logprob)
        loss = -torch.min(surr1, surr2)
        return loss

    def get_grad_norm(self, mini_batch):
        # self.policy.optimizer.zero_grad()
        # with torch.autograd.detect_anomaly():
        loss = self.calculate_loss(mini_batch)
        loss.mean().backward()
        # self.policy.optimizer.step()

        metric_array = []
        modules = [i for i in self.policy.policy_old.modules()]
        actor = modules[0].actor
        for layer in actor.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                metric_array.append(self.metric(layer))

        sum = 0.
        for i in range(len(metric_array)):
            sum += torch.sum(metric_array[i])
        return sum.item()

    
    def metric(self, layer):
        if layer.weight.grad is not None:
            return layer.weight.grad.norm()
        else:
            return torch.zeros_like(layer.weight)