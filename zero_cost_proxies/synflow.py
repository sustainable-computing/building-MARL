import numpy as np
import torch
import torch.nn as nn


class SynFlow():
    def __init__(self, policy):
        self.policy = policy

    def get_synflow(self):
        @torch.no_grad()
        def linearize(net):
            signs = {}
            for name, param in net.state_dict().items():
                signs[name] = torch.sign(param)
                param.abs_()
            return signs

        #convert to orig values
        @torch.no_grad()
        def nonlinearize(net, signs):
            for name, param in net.state_dict().items():
                if 'weight_mask' not in name:
                    param.mul_(signs[name])

        # keep signs of all params
        signs = linearize(self.policy.policy_old)
        
        # Compute gradients with input of 1s 
        self.policy.policy_old.zero_grad()
        self.policy.policy_old.double()
        input_dim = self.policy.state_dim
        inputs = torch.ones(input_dim).double()
        _, logprob = self.policy.policy_old.act(inputs, detach=False)
        torch.sum(logprob).backward()
        
        modules = [i for i in self.policy.policy_old.modules()]
        actor = modules[0].actor
        grads_abs = self.get_metric_arr(actor)

        nonlinearize(self.policy.policy_old, signs)

        sum = 0.
        for i in range(len(grads_abs)):
            sum += torch.sum(grads_abs[i])

        return sum.item()

    def get_metric_arr(self, net):
        metric_array = []

        for layer in net.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                metric_array.append(self.calc_synflow(layer))
        return metric_array

    
    def calc_synflow(self, layer):
        if layer.weight.grad is not None:
            return torch.abs(layer.weight * layer.weight.grad)
        else:
            return torch.zeros_like(layer.weight)