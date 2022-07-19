import argparse
import glob
import json
import os
import pickle
import random
from pkgutil import extend_path

import numpy as np
import pandas as pd
import torch
import tqdm

from ope.fittedq import FittedQ
from ppo import PPO


def init_parser():
    parser = argparse.ArgumentParser(description="Run UBC Policy Selection for specified zone")
    parser.add_argument("--zone", type=str, required=True)
    return parser


def get_policies(exclusion_list=None):
    invalid_policy_loc = "data/invalid_policy_list.json"
    with open(invalid_policy_loc, "r") as f:
        invalid_policies = json.load(f)["invalid_policies"]
    
    invalid_policies = [policy[3:] for policy in invalid_policies]

    all_policies = sorted(list(glob.glob(f"policy_library/**.pth")))

    valid_policies = list(set(all_policies) - set(invalid_policies))

    valid_policies = sorted(remove_env_diversity(valid_policies))

    if exclusion_list is not None:
        for policy in exclusion_list:
            valid_policies.remove(policy)

    init_valid_policies = init_policies(valid_policies)
    return valid_policies, init_valid_policies


def remove_env_diversity(policies):
    new_policies = []
    for policy in policies:
        if "blind" not in policy:
            new_policies.append(policy)
    return new_policies


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
        agent.policy_evaluation = False
        agent.policy_old.set_action_std(0.1)
        agents.append(agent)
    return agents


if __name__ == "__main__":
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    parser = init_parser()
    # args = parser.parse_args()

    # exclusion_list = [
    #     "policy_library/100_0.pth",
    #     "policy_library/100_1.pth",
    #     "policy_library/100_1_1e-1.pth",
    #     "policy_library/100_1_1e0.pth",
    #     "policy_library/100_1_1e1.pth",
    #     "policy_library/100_2.pth",
    #     "policy_library/100_3.pth",
    #     "policy_library/100_4.pth",
    #     "policy_library/100_4_1e0.pth",
    #     "policy_library/100_4_1e1.pth",
    #     "policy_library/101_0.pth",
    #     "policy_library/101_1.pth",
    #     "policy_library/101_1_1e-1.pth",
    #     "policy_library/101_2.pth",
    #     "policy_library/101_2_1e-1.pth",
    #     "policy_library/101_3.pth",
    #     "policy_library/101_3_1e1.pth",
    #     "policy_library/101_4.pth",
    #     "policy_library/102_0.pth",
    #     "policy_library/102_0_1e-1.pth"]

    policy_locs, init_policies = get_policies()
    test_zones = ['Core_top', 'Core_mid', 'Core_bottom',
                  'Perimeter_top_ZN_3', 'Perimeter_top_ZN_2', 'Perimeter_top_ZN_1', 'Perimeter_top_ZN_4',
                  'Perimeter_bot_ZN_3', 'Perimeter_bot_ZN_2', 'Perimeter_bot_ZN_1', 'Perimeter_bot_ZN_4',
                  'Perimeter_mid_ZN_3', 'Perimeter_mid_ZN_2', 'Perimeter_mid_ZN_1', 'Perimeter_mid_ZN_4']
    
    # zone = args.zone
    zone = "Perimeter_bot_ZN_3"
    if zone not in test_zones:
        raise ValueError("test_zone not valid")
    print(f"Running FQE for Zone {zone}")
    log_data = pd.read_csv("data/rule_based_log_data/0_cleaned_log.csv")
    # with open("data/rule_based_log_data/action_probs_all_data.pkl", "rb") as f:
    #     behavior_model = pickle.load(f)

    num_ts_per_day = 4 * 24
    num_days = 30
    ts_end = num_ts_per_day * num_days
    mini_batch = log_data[log_data["zone"] == zone].sort_values(by=["timestep"])
    mini_batch = log_data[:ts_end]

    log_dir_root = f"data/fqe_data/{zone}/"
    os.makedirs(log_dir_root, exist_ok=True)
    if len(os.listdir(log_dir_root)) == 0:
        run_num = 0
    else:
        run_num = int(sorted(os.listdir(log_dir_root))[-1]) + 1
    
    log_dir = os.path.join(log_dir_root, str(run_num))
    os.makedirs(log_dir, exist_ok=True)
    checkpoint_root = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoint_root, exist_ok=True)
    total = len(init_policies)
    pbar = tqdm.tqdm(total=total)
    # i = 0
    for policy_loc, policy in zip(policy_locs, init_policies):
        policy_name = policy_loc[15:-4]
        os.makedirs(os.path.join(log_dir, policy_name))
        qf = FittedQ(mini_batch, 6, 1, critic_lr=3e-4, weight_decay=1e-5, tau=0.005)
        qf.train_fitted_q(policy.select_action, log_loss=f"{log_dir}/{policy_name}/qfitted_loss.csv", epochs=600, p_bar=False)
        qf.save_params(checkpoint_root, policy_name)
    
        returns = qf.estimate_returns(policy.select_action, mini_batch)
        with open(f"{log_dir}/policy_returns.csv", "a+") as f:
            f.write(f"{policy_loc},{returns}\n")
        pbar.update(1)
        # i += 1
        # print(f"{i}/{total} policies evaluated")
