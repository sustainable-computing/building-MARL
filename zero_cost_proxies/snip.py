import torch
import torch.nn as nn
import torch.nn.functional as F
import types
from ope.iw import InverseProbabilityWeighting


class SNIP():
    def __init__(self, policy, behavior_policy):
        self.policy = policy
        self.behavior_policy = behavior_policy
    
    def snip_forward_linear(self, layer, x):
        return F.linear(x, layer.weight * layer.weight_mask, layer.bias)
    
    def snip(self, layer):
        if layer.weight_mask.grad is not None:
            return torch.abs(layer.weight_mask.grad)
        else:
            return torch.zeros_like(layer.weight)
    
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

    def compute_snip(self, mini_batch):
        modules = [i for i in self.policy.policy_old.modules()]
        actor = modules[0].actor
        for layer in actor.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
                layer.weight.requires_grad = False
            # Override the forward methods:
            # if isinstance(layer, nn.Conv2d):
            #     layer.forward = types.MethodType(snip_forward_conv2d, layer)

            if isinstance(layer, nn.Linear):
                layer.forward = types.MethodType(self.snip_forward_linear, layer)

        self.policy.policy_old.zero_grad()
        loss = self.calculate_loss(mini_batch)
        loss.mean().backward()

        metric_array = []
        for layer in actor.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                metric_array.append(self.snip(layer))
        
        sum = 0.
        for i in range(len(metric_array)):
            sum += torch.sum(metric_array[i])
        return sum.item()
