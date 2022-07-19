# implementing OPE of the IPWLearner using synthetic bandit data
from sklearn.linear_model import LogisticRegression
# import open bandit pipeline (obp)
from obp.policy import ContinuousNNPolicyLearner
from obp.ope import (
    ContinuousOffPolicyEvaluation,
    KernelizedInverseProbabilityWeighting,
    KernelizedDoublyRobust,
    KernelizedSelfNormalizedInverseProbabilityWeighting,

)
import pandas as pd
import pickle
from ppo import PPO
import torch
import numpy as np


log_data = pd.read_csv("data/rule_based_log_data/0_cleaned_log.csv")
num_ts_per_day = 4 * 24
num_days = 30
ts_end = num_ts_per_day * num_days
zones = log_data["zone"].unique()
zone = zones[0]
mini_batch = log_data[log_data["zone"] == zone].sort_values(by=["timestep"])
mini_batch = log_data[:ts_end]
print(zone)

policy = "policy_library/100_0.pth"
agent = PPO(6, 1, 0.003, 0.0005, 1, 10, 0.2,
                    has_continuous_action_space=True, action_std_init=0.2, 
                    device=torch.device('cpu'), diverse_policies=list(),
                    diverse_weight=0, diverse_increase=True)
agent.load(policy)
agent.policy_evaluation = False
agent.policy_old.set_action_std(0.1)

states = []
actions = []
rewards = []
for i, row in mini_batch.iterrows():
    state_vars = ["outdoor_temp", "solar_irradiation", "time_hour",
                  "zone_humidity", "zone_temp", "zone_occupancy"]
    state = [row[var] for var in state_vars]
    action = row["action"]
    reward = row["reward"]
    states.append(state)
    rewards.append(reward)
    actions.append(action)

eval_actions = torch.Tensor(agent.select_action(states)).sigmoid()
probs = torch.exp(agent.buffer.logprobs[0].reshape(-1, 1))
# (1) Generate Synthetic Bandit Data
# dataset = SyntheticBanditDataset(n_actions=10, reward_type="binary")
# bandit_feedback_train = dataset.obtain_batch_bandit_feedback(n_rounds=1000)
# bandit_feedback_test = dataset.obtain_batch_bandit_feedback(n_rounds=1000)

# (2) Off-Policy Learning
# eval_policy = IPWLearner(n_actions=1, base_classifier=LogisticRegression())
# eval_policy.fit(
#     context=states,
#     action=actions,
#     reward=rewards
#     # pscore=bandit_feedback_train["pscore"]
# )
# action_dist = eval_policy.predict(context=bandit_feedback_test["context"])

# (3) Off-Policy Evaluation
# `regression_model = RegressionModel(
#     n_actions=dataset.n_actions,
#     base_model=LogisticRegression(),
# )
# estimated_rewards_by_reg_model = regression_model.fit_predict(
#     context=bandit_feedback_test["context"],
#     action=bandit_feedback_test["action"],
#     reward=bandit_feedback_test["reward"],
# )
# ope = OffPolicyEvaluation(
#     bandit_feedback=bandit_feedback_test,
#     ope_estimators=[IPW()]
# )
# ope.visualize_off_policy_estimates(
#     action_dist=action_dist
#     # estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
# )

ope = ContinuousOffPolicyEvaluation(bandit_feedback={"action": np.array(actions),
                                                     "reward": np.array(rewards),
                                                     "pscore": np.ones((len(mini_batch)))},
                                    ope_estimators=[KernelizedInverseProbabilityWeighting(kernel="epanechnikov", bandwidth=0.02)])

estimated_val = ope.estimate_policy_values(action_by_evaluation_policy=eval_actions.numpy())
