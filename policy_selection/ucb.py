
import config
from cobs import Model

Model.set_energyplus_folder(config.energy_plus_loc)
import itertools
import os
import random
import traceback
from datetime import datetime
import json

import numpy as np


class UCB():
    def __init__(self, test_zone=None, log_dir="data/ucb_log_data/"):
        self.test_zone = test_zone
        self.log_dir = log_dir
        if not os.path.isdir(self.log_dir):
            self.run_num = 0
        else:
            runs = os.listdir(self.log_dir)
            runs = sorted([int(run) for run in runs])
            if len(runs):
                self.run_num = int(runs[-1]) + 1
            else:
                self.run_num = 0
    
    def make_log_dir(self):
        self.save_dir = os.path.join(self.log_dir, f"{self.run_num}")
        os.makedirs(self.save_dir)

    def make_cobs_model(self):
        available_zones = ['TopFloor_Plenum', 'MidFloor_Plenum', 'FirstFloor_Plenum',
                        'Core_top', 'Core_mid', 'Core_bottom',
                        'Perimeter_top_ZN_3', 'Perimeter_top_ZN_2', 'Perimeter_top_ZN_1', 'Perimeter_top_ZN_4',
                        'Perimeter_bot_ZN_3', 'Perimeter_bot_ZN_2', 'Perimeter_bot_ZN_1', 'Perimeter_bot_ZN_4',
                        'Perimeter_mid_ZN_3', 'Perimeter_mid_ZN_2', 'Perimeter_mid_ZN_1', 'Perimeter_mid_ZN_4']
        
        airloops = {'Core_top': "PACU_VAV_top", 'Core_mid': "PACU_VAV_mid", 'Core_bottom': "PACU_VAV_bot",
                    'Perimeter_top_ZN_3': "PACU_VAV_top", 'Perimeter_top_ZN_2': "PACU_VAV_top", 'Perimeter_top_ZN_1': "PACU_VAV_top", 'Perimeter_top_ZN_4': "PACU_VAV_top",
                    'Perimeter_bot_ZN_3': "PACU_VAV_bot", 'Perimeter_bot_ZN_2': "PACU_VAV_bot", 'Perimeter_bot_ZN_1': "PACU_VAV_bot", 'Perimeter_bot_ZN_4': "PACU_VAV_bot",
                    'Perimeter_mid_ZN_3': "PACU_VAV_mid", 'Perimeter_mid_ZN_2': "PACU_VAV_mid", 'Perimeter_mid_ZN_1': "PACU_VAV_mid", 'Perimeter_mid_ZN_4': "PACU_VAV_mid"}

        # Add state variables that we care about
        eplus_extra_states = {("Zone Air Relative Humidity", zone): f"{zone} humidity" for zone in available_zones}
        eplus_extra_states.update({("Heating Coil Electric Energy", f"{zone} VAV Box Reheat Coil"): f"{zone} vav energy" for zone in available_zones})
        eplus_extra_states.update({("Air System Electric Energy", airloop): f"{airloop} energy" for airloop in set(airloops.values())})
        eplus_extra_states[('Site Outdoor Air Drybulb Temperature', 'Environment')] = "outdoor temperature"
        eplus_extra_states[('Site Direct Solar Radiation Rate per Area', 'Environment')] = "site solar radiation"
        eplus_extra_states[('Facility Total HVAC Electric Demand Power', 'Whole Building')] = "total hvac"

        model = Model(idf_file_name="./eplus_files/OfficeMedium_Denver.idf",
                           weather_file="./eplus_files/USA_CO_Denver-Aurora-Buckley.AFB_.724695_TMY3.epw",
                           eplus_naming_dict=eplus_extra_states,
                           tmp_idf_path=self.save_dir)
        for key, _ in eplus_extra_states.items():
            model.add_configuration("Output:Variable",
                                    {"Key Value": key[1], "Variable Name": key[0], "Reporting Frequency": "Timestep"})
        
        # Setup controls to all VAV boxes
        control_zones = available_zones[3:]
        for zone in control_zones:
            model.add_configuration("Schedule:Constant",
                                    {"Name": f"{zone} VAV Customized Schedule",
                                    "Schedule Type Limits Name": "Fraction",
                                    "Hourly Value": 0})
            model.edit_configuration(idf_header_name="AirTerminal:SingleDuct:VAV:Reheat",
                                    identifier={"Name": f"{zone} VAV Box Component"},
                                    update_values={"Zone Minimum Air Flow Input Method": "Scheduled",
                                                   "Minimum Air Flow Fraction Schedule Name": f"{zone} VAV Customized Schedule"})
        self.model = model
    
    def set_start_date(self, run_duration=30, start_year=1994, start_month=1, start_day=1, timestep_per_hour=4):
        self.model.set_runperiod(days=run_duration, start_year=start_year, start_month=start_month, start_day=start_day, specify_year=True)
        self.model.set_timestep(timestep_per_hour=timestep_per_hour)
    
    def play(self, policy, eval_duration=30):
        possible_years = [1994, 1997, 1998, 1999, 2002, 2003, 2004, 2005]
        possible_months = range(12)
        possible_dates = range(31)

        year = random.choice(possible_years)
        month = random.choice(possible_months) + 1
        
        while True:
            day = random.choice(possible_dates) + 1
            try:
                date = datetime(year=year, month=month, day=day)
                if date.weekday() <= 4:
                    # Date is a weekday
                    break
            except ValueError:
                continue
        
        self.set_start_date(run_duration=eval_duration, start_year=year, start_month=month, start_day=day)
        state = self.model.reset()
        total_energy = state["total hvac"]
        while not self.model.is_terminate():
            occupancy = 1 if state["occupancy"][self.test_zone] > 0 else 0
            # Transfer the state into the format of only selected states
            action = policy.select_action([state["outdoor temperature"],
                                          state["site solar radiation"],
                                          state["time"].hour,
                                          state[f"{self.test_zone} humidity"],
                                          state["temperature"][self.test_zone],
                                          occupancy])

            action = np.array(action)
            action = 1/(1 + np.exp(-action))

            actions = list()
            actions.append({"priority": 0,
                            "component_type": "Schedule:Constant",
                            "control_type": "Schedule Value",
                            "actuator_key": f"{self.test_zone} VAV Customized Schedule",
                            "value": action,
                            "start_time": state['timestep'] + 1})
            state = self.model.step(actions)
            total_energy += state["total hvac"]

        # return -(total_energy - 5032951.13628954)/(6472063.181046309 - 5032951.13628954), year, month, date.day
        return -(total_energy - 14000)/(7e6 - 14000), year, month, date.day

        # return -total_energy/1e7, year, month, date.day

    def log_data(self, policy_name=None, policy_scores=None, policy_counts=None, flops=None,
                 initialize=False, policy_names=None, buffer_size=1,
                 start_year=None, start_month=None, start_day=None):
        if initialize:
            cols = ["datetime", "flops", "policy_name", "start_year",
                    "start_month", "start_day"]
            self.csv_loc = f"{self.save_dir}/ucb_log_data.csv"
            self.csv_file_obj = open(self.csv_loc, "w+", buffering=buffer_size)
            for policy in policy_names:
                cols.append(f"{policy}_score")
                cols.append(f"{policy}_count")
            csv_header = ",".join(cols)
            
            self.csv_file_obj.write(csv_header+"\n")
        else:
            row_data = [str(datetime.now()), str(flops), policy_name,
                        str(start_year), str(start_month), str(start_day)]
            for policy_idx in range(len(policy_names)):
                row_data.append(str(policy_scores[policy_idx]))
                row_data.append(str(policy_counts[policy_idx]))
            row_str = ",".join(row_data)
            row_str += "\n"
            self.csv_file_obj.write(row_str)

    def calc_ucb_value(self, policy_count, total_count, rho=2):
        if total_count == 0 or policy_count == 0:
            return 0
        else:
            return np.sqrt(rho*np.log(total_count)/policy_count)

    def run_ucb(self, policy_names=[], policies=[], epochs=None,
                policy_scores=None, policy_counts=None, rho=2, eval_duration=30):
        self.make_log_dir()
        self.log_data(initialize=True, policy_names=policy_names)
        self.make_cobs_model()

        if policy_scores is None:
            policy_scores = np.ones((len(policies))) * np.inf

        policy_scores_all = {}
        for policy in policy_names:
            policy_scores_all[policy] = []

        if policy_counts is None:
            policy_counts = np.zeros(len(policies))

        epoch = 0
        try:
            while True:
                print(f"EPOCH {epoch}\n")

                ucb_values = np.array([self.calc_ucb_value(policy_count, np.sum(policy_counts), rho=rho) for policy_count in policy_counts])

                chosen_policy_idx = np.argmax(policy_scores + ucb_values)
                policy_name = policy_names[chosen_policy_idx]
                q_policy, start_year, start_month, start_day = self.play(policies[chosen_policy_idx], eval_duration=eval_duration)

                if policy_counts[chosen_policy_idx] == 0:
                    policy_scores[chosen_policy_idx] = q_policy
                    policy_scores_all[policy_name].append(q_policy)
                    policy_counts[chosen_policy_idx] += 1
                else:
                    policy_scores_all[policy_name].append(q_policy)
                    policy_counts[chosen_policy_idx] += 1
                    policy_mu = np.mean(policy_scores_all[policy_name])
                    policy_scores[chosen_policy_idx] = policy_mu
                epoch += 1
                self.log_data(policy_scores=policy_scores, policy_counts=policy_counts, flops=0, policy_names=policy_names,
                              start_year=start_year, start_month=start_month, start_day=start_day, policy_name=policy_name)
                if epochs is not None:
                    if epoch == epochs:
                        self.csv_file_obj.close()
                        break
        finally:
            traceback.print_exc()
            self.csv_file_obj.close()


class GroupedUCB():
    def __init__(self, group_config, policy_locs, init_policies, log_dir):
        self.group_config = group_config
        self.log_dir = log_dir
        self.policies = {}
        for i, policy_loc in enumerate(policy_locs):
            self.policies[policy_loc] = init_policies[i]

        if not os.path.isdir(self.log_dir):
            self.run_num = 0
        else:
            runs = os.listdir(self.log_dir)
            runs = sorted([int(run) for run in runs])
            if len(runs):
                self.run_num = int(runs[-1]) + 1
            else:
                self.run_num = 0
        
        self.init_arms()
    
    def init_arms(self):
        groups = list(self.group_config["groups"].keys())
        policies = []
        for group in groups:
            policies.append(self.group_config["group_policies"][group])
        combinations = itertools.product(*policies)
        self.arms = {}
        for i, combination in enumerate(combinations):
            arm = {}
            for j, value in enumerate(combination):
                arm[groups[j]] = value
            self.arms[i] = arm
        self.group_config["arms"] = self.arms

    def make_log_dir(self):
        self.save_dir = os.path.join(self.log_dir, f"{self.run_num}")
        os.makedirs(self.save_dir)
        with open(os.path.join(self.save_dir, "group_config.json"), "w+") as f:
            json.dump(self.group_config, f)

    def make_cobs_model(self):
        available_zones = ['TopFloor_Plenum', 'MidFloor_Plenum', 'FirstFloor_Plenum',
                        'Core_top', 'Core_mid', 'Core_bottom',
                        'Perimeter_top_ZN_3', 'Perimeter_top_ZN_2', 'Perimeter_top_ZN_1', 'Perimeter_top_ZN_4',
                        'Perimeter_bot_ZN_3', 'Perimeter_bot_ZN_2', 'Perimeter_bot_ZN_1', 'Perimeter_bot_ZN_4',
                        'Perimeter_mid_ZN_3', 'Perimeter_mid_ZN_2', 'Perimeter_mid_ZN_1', 'Perimeter_mid_ZN_4']
        
        airloops = {'Core_top': "PACU_VAV_top", 'Core_mid': "PACU_VAV_mid", 'Core_bottom': "PACU_VAV_bot",
                    'Perimeter_top_ZN_3': "PACU_VAV_top", 'Perimeter_top_ZN_2': "PACU_VAV_top", 'Perimeter_top_ZN_1': "PACU_VAV_top", 'Perimeter_top_ZN_4': "PACU_VAV_top",
                    'Perimeter_bot_ZN_3': "PACU_VAV_bot", 'Perimeter_bot_ZN_2': "PACU_VAV_bot", 'Perimeter_bot_ZN_1': "PACU_VAV_bot", 'Perimeter_bot_ZN_4': "PACU_VAV_bot",
                    'Perimeter_mid_ZN_3': "PACU_VAV_mid", 'Perimeter_mid_ZN_2': "PACU_VAV_mid", 'Perimeter_mid_ZN_1': "PACU_VAV_mid", 'Perimeter_mid_ZN_4': "PACU_VAV_mid"}

        # Add state variables that we care about
        eplus_extra_states = {("Zone Air Relative Humidity", zone): f"{zone} humidity" for zone in available_zones}
        eplus_extra_states.update({("Heating Coil Electric Energy", f"{zone} VAV Box Reheat Coil"): f"{zone} vav energy" for zone in available_zones})
        eplus_extra_states.update({("Air System Electric Energy", airloop): f"{airloop} energy" for airloop in set(airloops.values())})
        eplus_extra_states[('Site Outdoor Air Drybulb Temperature', 'Environment')] = "outdoor temperature"
        eplus_extra_states[('Site Direct Solar Radiation Rate per Area', 'Environment')] = "site solar radiation"
        eplus_extra_states[('Facility Total HVAC Electric Demand Power', 'Whole Building')] = "total hvac"

        model = Model(idf_file_name="./eplus_files/OfficeMedium_Denver.idf",
                           weather_file="./eplus_files/USA_CO_Denver-Aurora-Buckley.AFB_.724695_TMY3.epw",
                           eplus_naming_dict=eplus_extra_states,
                           tmp_idf_path=self.save_dir)
        for key, _ in eplus_extra_states.items():
            model.add_configuration("Output:Variable",
                                    {"Key Value": key[1], "Variable Name": key[0], "Reporting Frequency": "Timestep"})
        
        # Setup controls to all VAV boxes
        control_zones = available_zones[3:]
        for zone in control_zones:
            model.add_configuration("Schedule:Constant",
                                    {"Name": f"{zone} VAV Customized Schedule",
                                    "Schedule Type Limits Name": "Fraction",
                                    "Hourly Value": 0})
            model.edit_configuration(idf_header_name="AirTerminal:SingleDuct:VAV:Reheat",
                                    identifier={"Name": f"{zone} VAV Box Component"},
                                    update_values={"Zone Minimum Air Flow Input Method": "Scheduled",
                                                   "Minimum Air Flow Fraction Schedule Name": f"{zone} VAV Customized Schedule"})
        self.model = model
    
    def set_start_date(self, run_duration=30, start_year=1994, start_month=1, start_day=1, timestep_per_hour=4):
        self.model.set_runperiod(days=run_duration, start_year=start_year, start_month=start_month, start_day=start_day, specify_year=True)
        self.model.set_timestep(timestep_per_hour=timestep_per_hour)

    def play(self, arm, eval_duration=30):
        possible_years = [1994, 1997, 1998, 1999, 2002, 2003, 2004, 2005]
        possible_months = range(12)
        possible_dates = range(31)

        year = random.choice(possible_years)
        month = random.choice(possible_months) + 1
        
        while True:
            day = random.choice(possible_dates) + 1
            try:
                date = datetime(year=year, month=month, day=day)
                if date.weekday() <= 4:
                    # Date is a weekday
                    break
            except ValueError:
                continue
        
        self.set_start_date(run_duration=eval_duration, start_year=year, start_month=month, start_day=day)
        state = self.model.reset()
        total_energy = state["total hvac"]
        while not self.model.is_terminate():
            action = list()
            for group in arm.keys():
                policy = self.policies[arm[group]]
                for zone in self.group_config["groups"][group]:
                    occupancy = 1 if state["occupancy"][zone] > 0 else 0
                    zone_state = [state["outdoor temperature"], state["site solar radiation"],
                                  state["time"].hour, state[f"{zone} humidity"],
                                  state["temperature"][zone], occupancy]
                    zone_action = policy.select_action(zone_state)
                    zone_action = np.array(zone_action)
                    zone_action = 1/(1 + np.exp(-zone_action))
                    
                    action.append({"priority": 0,
                            "component_type": "Schedule:Constant",
                            "control_type": "Schedule Value",
                            "actuator_key": f"{zone} VAV Customized Schedule",
                            "value": zone_action,
                            "start_time": state['timestep'] + 1})

            state = self.model.step(action)
            total_energy += state["total hvac"]

        # return -(total_energy - 5032951.13628954)/(6472063.181046309 - 5032951.13628954), year, month, date.day
        # return -(total_energy - 14000)/(7e6 - 14000), year, month, date.day
        return - (total_energy - 61540255.66407317) / (613290612.4874102 - 61540255.66407317), year, month, date.day, total_energy  # One month eval (30 days)

    def log_data(self, arm_name=None, arm_scores=None, arm_counts=None, flops=None,
                 initialize=False, arm_names=None, buffer_size=1,
                 start_year=None, start_month=None, start_day=None,
                 total_energy=None):
        if initialize:
            cols = ["datetime", "flops", "policy_name", "start_year",
                    "start_month", "start_day", "total_energy"]
            self.csv_loc = os.path.join(self.save_dir, "ucb_log_data.csv")
            self.csv_file_obj = open(self.csv_loc, "w+", buffering=buffer_size)
            for arm in self.arms:
                cols.append(f"arm_{arm}_score")
                cols.append(f"arm_{arm}_count")
            csv_header = ",".join(cols)
            
            self.csv_file_obj.write(csv_header+"\n")
        else:
            row_data = [str(datetime.now()), str(flops), str(arm_name),
                        str(start_year), str(start_month), str(start_day),
                        str(total_energy)]
            for policy_idx in range(len(arm_names)):
                row_data.append(str(arm_scores[policy_idx]))
                row_data.append(str(arm_counts[policy_idx]))
            row_str = ",".join(row_data)
            row_str += "\n"
            self.csv_file_obj.write(row_str)

    def calc_ucb_value(self, policy_count, total_count, rho=2):
        if total_count == 0 or policy_count == 0:
            return 0
        else:
            return np.sqrt(rho*np.log(total_count)/policy_count)

    def run_ucb(self, epochs=None, arm_scores=None,
                arm_counts=None, rho=2, eval_duration=30):
        self.make_log_dir()
        self.log_data(initialize=True)
        self.make_cobs_model()

        if arm_scores is None:
            arm_scores = np.ones((len(self.arms))) * np.inf

        arm_scores_all = {}
        for arm in self.arms:
            arm_scores_all[arm] = []

        if arm_counts is None:
            arm_counts = np.zeros(len(self.arms))

        epoch = 0
        try:
            while True:
                print(f"EPOCH {epoch}\n")

                ucb_values = np.array([self.calc_ucb_value(arm_count, np.sum(arm_counts), rho=rho) for arm_count in arm_counts])

                chosen_arm_idx = np.argmax(arm_scores + ucb_values)
                arm_name = list(self.arms.keys())[chosen_arm_idx]
                q_policy, start_year, start_month, start_day, total_energy = self.play(self.arms[chosen_arm_idx], eval_duration=eval_duration)

                if arm_counts[chosen_arm_idx] == 0:
                    arm_scores[chosen_arm_idx] = q_policy
                    arm_scores_all[arm_name].append(q_policy)
                    arm_counts[chosen_arm_idx] += 1
                else:
                    arm_scores_all[arm_name].append(q_policy)
                    arm_counts[chosen_arm_idx] += 1
                    policy_mu = np.mean(arm_scores_all[arm_name])
                    arm_scores[chosen_arm_idx] = policy_mu
                epoch += 1
                self.log_data(arm_scores=arm_scores, arm_counts=arm_counts, flops=0, arm_names=list(self.arms.keys()),
                              start_year=start_year, start_month=start_month, start_day=start_day, arm_name=arm_name,
                              total_energy=total_energy)
                if epochs is not None:
                    if epoch == epochs:
                        self.csv_file_obj.close()
                        break
        finally:
            traceback.print_exc()
            self.csv_file_obj.close()
