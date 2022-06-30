import glob
import json

import torch

from policy_selection.ucb import UCB
from ppo import PPO


def get_policies():
    invalid_policy_loc = "data/invalid_policy_list.json"
    with open(invalid_policy_loc, "r") as f:
        invalid_policies = json.load(f)["invalid_policies"]
    
    invalid_policies = [policy[3:] for policy in invalid_policies]

    all_policies = sorted(list(glob.glob(f"policy_library/**.pth")))

    valid_policies = list(set(all_policies) - set(invalid_policies))

    valid_policies = sorted(remove_env_diversity(valid_policies))
    init_valid_policies = init_policies(valid_policies)
    return valid_policies, init_valid_policies


def init_policies(policies):
    agents = []
    device = torch.device('cpu')

    for policy_loc in policies:
        agent = PPO(1 + 1 + 1 + 1 + 1 + 1, # State dimension, own temperature + humidity + outdoor temp + solar + occupancy + hour
                    1, # Action dimension, 1 for each zone
                    0.003, 0.0005, 1, 10, 0.2, has_continuous_action_space=True, action_std_init=0.2, 
                    device=device,
                    diverse_policies=list(), diverse_weight=0, diverse_increase=True)
        agent.load(policy_loc)
        agents.append(agent)
    return agents

def remove_env_diversity(policies):
    new_policies = []
    for policy in policies:
        if "blind" not in policy:
            new_policies.append(policy)
    return new_policies

def run_ucb(test_zone, policy_locs, policies, rho=2, eval_duration=30, epochs=100):
    ucb = UCB(test_zone, log_dir=f"data/ucb_log_data/{test_zone}/")
    ucb.run_ucb(policy_names=policy_locs, policies=policies, epochs=epochs, rho=rho, eval_duration=eval_duration)



if __name__ == "__main__":
    policy_locs, init_policies = get_policies()
    test_zones = ['Core_top', 'Core_mid', 'Core_bottom',
                  'Perimeter_top_ZN_3', 'Perimeter_top_ZN_2', 'Perimeter_top_ZN_1', 'Perimeter_top_ZN_4',
                  'Perimeter_bot_ZN_3', 'Perimeter_bot_ZN_2', 'Perimeter_bot_ZN_1', 'Perimeter_bot_ZN_4',
                  'Perimeter_mid_ZN_3', 'Perimeter_mid_ZN_2', 'Perimeter_mid_ZN_1', 'Perimeter_mid_ZN_4']

    test_zone = test_zones[0]  # Choosing zone to run UCB on
    print(f"Running UCB for Zone {test_zone}")

    run_ucb(test_zone, policy_locs, init_policies, rho=2, eval_duration=1, epochs=None)
