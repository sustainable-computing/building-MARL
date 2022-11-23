import glob
import json

import torch

from policy_selection.ucb import GroupedUCB
from ppo import PPO
import sys


def get_policies(policies=None, policy_dir="policy_library/**.pth", rm_env_diversity=False):
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
                  use_dummy_arms=False, dummy_init=[], dummy_energy_init=[],
                  random_seed=None, environment="B_Denver"):
    # ucb = GroupedUCB(group_config, policy_locs, init_policies,
    #                  log_dir=f"data/group_ucb_log_data/",
    #                  pickup_from=pickup_from, use_dummy_arms=use_dummy_arms,
    #                  dummy_init=dummy_init, dummy_energy_init=dummy_energy_init)
    #               use_dummy_arms=False, random_seed=None):
    ucb = GroupedUCB(group_config, policy_locs, init_policies,
                     log_dir=f"data/group_ucb_log_data/",
                     pickup_from=pickup_from, use_dummy_arms=use_dummy_arms,
                     random_seed=random_seed, environment=environment)
    ucb.run_ucb(epochs=epochs, rho=rho, eval_duration=eval_duration)


if __name__ == "__main__":
    group_config_denver = {
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
            "perimeter_group": ["policy_library/105_4_1e0.pth",
                                "policy_library/102_1_1e-1.pth",
                                "policy_library/105_4_blind.pth"],
                                # "policy_library/115_1_1e-1_blind.pth",
                                # "policy_library/107_4_1e0_blind.pth"],
            "core_group": ["policy_library/105_4_1e0.pth",
                           "policy_library/102_1_1e-1.pth",
                           "policy_library/105_4_blind.pth"]
                        #    "policy_library/115_1_1e-1_blind.pth",
                        #    "policy_library/107_4_1e0_blind.pth"]
        }
    }

    group_config_san_francisco = {
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
            "perimeter_group": ["policy_library/105_4_1e0.pth",
                                "policy_library/115_1_1e-1_blind.pth",
                                "policy_library/106_3_blind.pth"],
                                # "policy_library/115_1_1e-1_blind.pth",
                                # "policy_library/107_4_1e0_blind.pth"],
            "core_group": ["policy_library/105_4_1e0.pth",
                           "policy_library/115_1_1e-1_blind.pth",
                           "policy_library/106_3_blind.pth"]
                        #    "policy_library/115_1_1e-1_blind.pth",
                        #    "policy_library/107_4_1e0_blind.pth"]
        }
    }

    group_config_c = {
        "groups": {
            "north_group": ["North-1", "North-2", "North-3", "North-G"],
            "south_group": ["South-1", "South-2", "South-3", "South-GF"],
            "misc_group": ["Amphitheater", "Lab", "Library"]
        },
        "group_policies": {
            "north_group": ["policy_library/118_4_1e1.pth",
                           "policy_library/107_4_1e0_blind.pth",
                           "policy_library/108_0_blind.pth"],
            "south_group": ["policy_library/118_4_1e1.pth",
                           "policy_library/107_4_1e0_blind.pth",
                           "policy_library/108_0_blind.pth"],
            "misc_group": ["policy_library/118_4_1e1.pth",
                           "policy_library/107_4_1e0_blind.pth",
                           "policy_library/108_0_blind.pth"]
        }
    }
    seed = int(sys.argv[1])
    environment = str(sys.argv[2])

    if environment == "B_Denver":
        group_config = group_config_denver
    elif environment == "B_SanFrancisco":
        group_config = group_config_san_francisco
    elif environment == "C":
        group_config = group_config_c
    else:
        raise ValueError("Environment not supported")

    all_group_policies = []
    for group in group_config["groups"]:
        all_group_policies += group_config["group_policies"][group]

    policy_locs, init_policies = get_policies(policies=all_group_policies, rm_env_diversity=False)
    print("Seed: ", seed, "Environment: ", environment)
    run_group_ucb(group_config, policy_locs, init_policies,
                  rho=2, eval_duration=30, epochs=None,
                  use_dummy_arms=False, random_seed=seed,
                  pickup_from=None, environment=environment)