import glob
import json

import torch

from policy_selection.ucb import GroupedUCB
from ppo import PPO
import sys


def get_policies(policies=None, policy_dir="policy_library_20220820/**.pth", rm_env_diversity=False):
    invalid_policy_loc = "data/invalid_policy_list.json"
    with open(invalid_policy_loc, "r") as f:
        invalid_policies = json.load(f)["invalid_policies"]

    invalid_policies = []

    if policies is None:
        all_policies = sorted(list(glob.glob(policy_dir)))
        valid_policies = list(set(all_policies) - set(invalid_policies))
    else:
        valid_policies = policies

    if rm_env_diversity:
        valid_policies = remove_env_diversity(valid_policies)

    init_valid_policies = init_policies(valid_policies)
    return valid_policies, init_valid_policies


def init_policies(policies):
    agents = []
    device = torch.device('cpu')

    for policy_loc in policies:
        agent = PPO(1 + 1 + 1 + 1 + 1 + 1,  # State dimension, own temperature + humidity + outdoor temp + solar + occupancy + hour
                    1,  # Action dimension, 1 for each zone
                    0.003, 0.0005, 1, 10, 0.2, has_continuous_action_space=True, action_std_init=0.2,
                    device=device,
                    diverse_policies=list(), diverse_weight=0, diverse_increase=True)
        agent.load(policy_loc)
        agents.append(agent)
        # agents.eval()
    return agents


def remove_env_diversity(policies):
    new_policies = []
    for policy in policies:
        if "blind" not in policy:
            new_policies.append(policy)
    return new_policies


def run_group_ucb(group_config, policy_locs, init_policies, rho=2,
                  eval_duration=30, epochs=100, pickup_from=None,
                  use_dummy_arms=False, random_seed=None):
    ucb = GroupedUCB(group_config, policy_locs, init_policies,
                     log_dir=f"data/group_ucb_log_data/",
                     pickup_from=pickup_from, use_dummy_arms=use_dummy_arms,
                     random_seed=random_seed)
    ucb.run_ucb(epochs=epochs, rho=rho, eval_duration=eval_duration)


if __name__ == "__main__":
    group_config = {
        "groups": {
            "perimeter_group": ['Perimeter_top_ZN_3', 'Perimeter_top_ZN_2',
                                'Perimeter_top_ZN_1', 'Perimeter_top_ZN_4',
                                'Perimeter_bot_ZN_3', 'Perimeter_bot_ZN_2',
                                'Perimeter_bot_ZN_1', 'Perimeter_bot_ZN_4',
                                'Perimeter_mid_ZN_3', 'Perimeter_mid_ZN_2',
                                'Perimeter_mid_ZN_1', 'Perimeter_mid_ZN_4'],
            "core_group": ['Core_top', 'Core_mid', 'Core_bottom']
        },
        "group_policies": {
            "perimeter_group": ["policy_library_20220820/105_4_1e0_2.pth",
                                "policy_library_20220820/102_1_1e-1_2.pth",
                                "policy_library_20220820/105_4_blind.pth"],
                                # "policy_library_20220820/115_1_1e-1_2_blind.pth",
                                # "policy_library_20220820/107_4_1e0_blind.pth"],
            "core_group": ["policy_library_20220820/105_4_1e0_2.pth",
                           "policy_library_20220820/102_1_1e-1_2.pth",
                           "policy_library_20220820/105_4_blind.pth"]
                        #    "policy_library_20220820/115_1_1e-1_2_blind.pth",
                        #    "policy_library_20220820/107_4_1e0_blind.pth"]
        }
    }

    all_group_policies = []
    for group in group_config["groups"]:
        all_group_policies += group_config["group_policies"][group]

    policy_locs, init_policies = get_policies(policies=all_group_policies, rm_env_diversity=False)
    # print(int(sys.argv[1])
    seed = int(sys.argv[1])
    pickup = int(sys.argv[2])
    run_group_ucb(group_config, policy_locs, init_policies,
                  rho=2, eval_duration=30, epochs=1000,
                  use_dummy_arms=False, random_seed=seed,
                  pickup_from=pickup)

"""
nohup python group_ucb_evaluation.py 1337> data/group_ucb_std_data/random_seed_1337.out &
nohup python group_ucb_evaluation.py 31720 13 > data/group_ucb_std_data/random_seed_31720.out &
nohup python group_ucb_evaluation.py 21538 14 > data/group_ucb_std_data/random_seed_21538.out &
nohup python group_ucb_evaluation.py 69175 15 > data/group_ucb_std_data/random_seed_69175.out &
nohup python group_ucb_evaluation.py 17170 16 > data/group_ucb_std_data/random_seed_17170.out &
nohup python group_ucb_evaluation.py 36621 17 > data/group_ucb_std_data/random_seed_36621.out &
nohup python group_ucb_evaluation.py 41408 18 > data/group_ucb_std_data/random_seed_41408.out &
nohup python group_ucb_evaluation.py 22808 19 > data/group_ucb_std_data/random_seed_22808.out &
nohup python group_ucb_evaluation.py 19197 20 > data/group_ucb_std_data/random_seed_19197.out &
nohup python group_ucb_evaluation.py 23509 21 > data/group_ucb_std_data/random_seed_23509.out &
nohup python group_ucb_evaluation.py 69266 22 > data/group_ucb_std_data/random_seed_69266.out &
nohup python group_ucb_evaluation.py 36621 23 > data/group_ucb_std_data/random_seed_36621.out &
"""