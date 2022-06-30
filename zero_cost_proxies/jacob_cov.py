import torch
import numpy as np


class JacobianCovariance():
    def __init__(self, policy):
        self.policy = policy

    def get_batch_jacobian(self, net, x):
        x.requires_grad_(True)

        action, logprob = net(x, detach=False)
        logprob.backward(torch.ones_like(logprob))

        jacob = x.grad.detach()
        x.requires_grad_(False)
        return jacob

    def eval_score(self, jacob):
        corrs = np.ma.corrcoef(jacob).data
        v, _  = np.linalg.eig(corrs)
        k = 1e-5
        return -np.sum(np.log(v + k) + 1./(v + k))

    def compute_jacob_cov(self, mini_batch):
        # Compute gradients (but don't apply them)
        self.policy.policy_old.zero_grad()

        jacobs = self.get_batch_jacobian(self.policy.policy_old.act, mini_batch)
        jacobs = jacobs.reshape(jacobs.size(0), -1).cpu().numpy()

        try:
            jc = self.eval_score(jacobs)
        except Exception as e:
            print(e)
            jc = np.nan

        return np.absolute(jc)
