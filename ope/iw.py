import numpy as np
from scipy.integrate import quad
import torch
import torchquad


class InverseProbabilityWeighting():
    """Inverse probability weighting

    Source code adapted from https://github.com/st-tech/zr-obp
    """

    def __init__(self, sars_data, retain_grad_fn=False, univariate_action=True):
        """Class constructor for the InverseProbabilityWeighting class

        Args:
            sars_data (pandas.DataFrame): The dataset
        """
        self.sars_data = sars_data
        self.retain_grad_fn = retain_grad_fn
        self.univariate_action = univariate_action

    def evaluate_policy(self, eval_action_model,
                        behavior_action_model, score="mean"):
        """Method to conduct offline policy evaluation

        Args:
            eval_action_model (torch.model): The trained actor model
            behavior_action_model (dict): The saved action model generated from log data
            score (str): String indicating which scoring metric to use. Default is mean
        """
        data = self.sars_data.to_dict("records")
        action_prob = torch.zeros((len(data)))
        rewards = torch.zeros((len(data)))

        behavior_prob = torch.zeros((len(data)))
        states = []

        for i, row in enumerate(data):
            state_vars = ["outdoor_temp", "solar_irradiation", "time_hour",
                          "zone_humidity", "zone_temp", "zone_occupancy"]
            state = [row[var] for var in state_vars]

            state_bins = [np.digitize(row[var], behavior_action_model[f"{var}_bins"])-1 for var in state_vars]
            action_bin = np.digitize(row["action"], behavior_action_model["action_bins"]) - 1
            state_bins_str = "{},{},{},{},{},{}".format(*state_bins)
            action_dist = eval_action_model(state, no_grad = not self.retain_grad_fn)
            action_prob[i] = self.calculate_action_probability(action_dist, action_bin, behavior_action_model["action_bins"])
            behavior_prob[i] = behavior_action_model[state_bins_str][action_bin] / behavior_action_model["total_count"]
            rewards[i] = row["reward"]
            states.append(state)
        
        iw = action_prob / behavior_prob
        ret_data = iw * rewards
        if score == "mean":
            return ret_data.mean(), states, rewards, action_prob, behavior_prob
        else:
            return ret_data, states, rewards, action_prob, behavior_prob
    
    def inv_sigmoid(self, value):
        return np.log(value / (1- value))

    def calculate_action_probability(self, dist, bin, action_bins):
        bin_l = self.inv_sigmoid(action_bins[bin])
        bin_r = self.inv_sigmoid(action_bins[bin+1])
        if self.univariate_action:
            integral = dist.cdf(torch.Tensor([bin_r])) - dist.cdf(torch.Tensor([bin_l]))
        else:
            func = lambda inp: dist.log_prob(inp).exp()
            if bin_l == -np.inf:
                bin_l = -10
            if bin_r == np.inf:
                bin_r = 10
            if self.retain_grad_fn:
                int_method = torchquad.Trapezoid()
                integral = int_method.integrate(func, dim=1, N=1000,
                                    integration_domain=[[bin_l, bin_r]], 
                                    backend="torch")
            else:
                integral, err = quad(func, bin_l, bin_r)
        return integral
