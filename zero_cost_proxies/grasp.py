import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from ope.iw import InverseProbabilityWeighting


class GRASP():
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
        surr1, surr2 = self.policy.calculate_loss(discounted_rewards, state_values, behavior_logprob, policy_logprob, log_probs=False)
        loss = -torch.min(surr1, surr2)
        return loss
    
    def compute_grasp(self, mini_batch, T=1, num_iters=1):
        modules = [i for i in self.policy.policy_old.modules()]
        actor = modules[0].actor

        weights = []
        for layer in actor.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                weights.append(layer.weight)
                layer.weight.requires_grad_(True)
        
        self.policy.policy_old.zero_grad()
        grad_w = None
        for _ in range(num_iters):
            #TODO get new data, otherwise num_iters is useless!
            # outputs = self.policy.policy_old.act(inputs, detach=False)/T
            # loss = loss_fn(outputs, targets[st:en])
            loss = self.calculate_loss(mini_batch)
            grad_w_p = autograd.grad(loss.mean(), weights, allow_unused=True)
            if grad_w is None:
                grad_w = list(grad_w_p)
            else:
                for idx in range(len(grad_w)):
                    grad_w[idx] += grad_w_p[idx]
            
        # outputs = net.forward(inputs[st:en])/T
        # loss = loss_fn(outputs, targets[st:en])
        loss = self.calculate_loss(mini_batch)
        grad_f = autograd.grad(loss.mean(), weights, create_graph=True, allow_unused=True)
        
        # accumulate gradients computed in previous step and call backwards
        z, count = 0,0
        for layer in actor.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                if grad_w[count] is not None:
                    z += (grad_w[count].data * grad_f[count]).sum()
                count += 1
        
        z.backward()

        metric_array = []
        for layer in actor.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                metric_array.append(self.grasp(layer))
        
        sum = 0.
        for i in range(len(metric_array)):
            sum += torch.sum(metric_array[i])
        return sum.item()

    def grasp(self, layer):
        if layer.weight.grad is not None:
            return -layer.weight.data * layer.weight.grad   # -theta_q Hg
            #NOTE in the grasp code they take the *bottom* (1-p)% of values
            #but we take the *top* (1-p)%, therefore we remove the -ve sign
            #EDIT accuracy seems to be negatively correlated with this metric, so we add -ve sign here!
        else:
            return torch.zeros_like(layer.weight)