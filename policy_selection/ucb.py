
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
import pandas as pd
import pickle

class GroupedUCB():
    """Class to execute GroupedUCB for policy selection in the COBS framework

    Attributes:
        group_config (dict): Dictionary containing the configuration for the environment
        policy_locs (list): List of strings containing the locations of the policies
        init_policies (list): List of strings containing the loaded policies to be used
        log_dir (str): Directory to save the logs
        pickup_from (int): Run number to continue GroupedUCB from if execution had stopped in middle
        use_dummy_arms (bool): Whether to use dummy arms or not (for testing purposes)
        random_seed (int): Random seed to be used for reproducibility
        dummy_init (list): List of tuples containing the initial values for the dummy arms (for testing purposes)
        dummy_energy_init (list): List of tuples containing the initial values for the dummy energy arms (for testing purposes)
        environment (str): Name of the environment to be used
        policies (dict): Dictionary containing the policies to be used where the key is the policy location, and the value is the loaded policy
    """
    def __init__(self, group_config, policy_locs, init_policies,
                 log_dir, pickup_from, use_dummy_arms=False, random_seed=None,
                 dummy_init=[], dummy_energy_init=[], environment="B_Denver"):
        self.group_config = group_config
        self.log_dir = log_dir
        self.policies = {}
        self.pickup_from = pickup_from

        self.environment = environment

        for i, policy_loc in enumerate(policy_locs):
            self.policies[policy_loc] = init_policies[i]
        if pickup_from is None:
            if not os.path.isdir(self.log_dir):
                self.run_num = 0
            else:
                runs = os.listdir(self.log_dir)
                runs = sorted([int(run) for run in runs])
                if len(runs):
                    self.run_num = int(runs[-1]) + 1
                else:
                    self.run_num = 0
        else:
            self.run_num = pickup_from
        
        self.init_arms()

        self.use_dummy_arms = use_dummy_arms
        if self.use_dummy_arms:
            self.init_dummy_arms(dummy_init, dummy_energy_init)
            self.init_dummy_arms()

        if random_seed is not None:
            np.random.seed(random_seed)
            import torch
            torch.manual_seed(random_seed)

    def init_arms(self):
        """Method to generate the arms containing all possible combinations of policies for each group
        """
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
    
    def init_dummy_arms(self, dummy_init=[], dummy_energy_init=[]):
        """Method to initialize the dummy arms with all possible combinations of policies for each group
        
        For testing purposes only

        Args:
            dummy_init (list, optional): The initial distribution of the arms. Defaults to [].
            dummy_energy_init (list, optional): The initial distribution of energy of the arms. Defaults to [].
        """
        self.dummy_arms = {}
        if len(dummy_init) == 0:
            for i, arm in enumerate(self.arms):
                self.dummy_arms[arm] = {}
                self.dummy_arms[arm]["sigma"] = 0.5
                self.dummy_arms[arm]["mu"] = i/4
        else:
            for i, arm in enumerate(self.arms):
                self.dummy_arms[arm] = {}
                self.dummy_arms[arm]["mu"] = dummy_init[i][0]
                self.dummy_arms[arm]["sigma"] = dummy_init[i][1]
        
        self.dummy_arms_energy = {}
        if len(dummy_energy_init) == 0:
            for i, arm in enumerate(self.arms):
                self.dummy_arms_energy[arm] = {}
                self.dummy_arms_energy[arm]["sigma"] = 0
                self.dummy_arms_energy[arm]["mu"] = 0
        else:
            for i, arm in enumerate(self.arms):
                self.dummy_arms_energy[arm] = {}
                self.dummy_arms_energy[arm]["mu"] = dummy_energy_init[i][0]
                self.dummy_arms_energy[arm]["sigma"] = dummy_energy_init[i][1]

    def make_log_dir(self):
        """Method to create the log directory for the current run"""
        self.save_dir = os.path.join(self.log_dir, f"{self.run_num}")
        os.makedirs(self.save_dir, exist_ok=True)
        with open(os.path.join(self.save_dir, "group_config.json"), "w+") as f:
            json.dump(self.group_config, f)

    def make_cobs_model(self):
        """Method to create the COBS model for the current run

        Raises:
            ValueError: If the specified environment is not supported
        """
        if self.environment.startswith("B_"):
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

            if self.environment == "B_Denver":
                model = Model(idf_file_name="./eplus_files/OfficeMedium_Denver.idf",
                                weather_file="./eplus_files/USA_CO_Denver-Aurora-Buckley.AFB_.724695_TMY3.epw",
                                eplus_naming_dict=eplus_extra_states,
                                tmp_idf_path=self.save_dir)
                for key, _ in eplus_extra_states.items():
                    model.add_configuration("Output:Variable",
                                            {"Key Value": key[1], "Variable Name": key[0], "Reporting Frequency": "Timestep"})
                
                # Setup controls to all VAV boxes
                self.control_zones = available_zones[3:]
                for zone in self.control_zones:
                    model.add_configuration("Schedule:Constant",
                                            {"Name": f"{zone} VAV Customized Schedule",
                                            "Schedule Type Limits Name": "Fraction",
                                            "Hourly Value": 0})
                    model.edit_configuration(idf_header_name="AirTerminal:SingleDuct:VAV:Reheat",
                                            identifier={"Name": f"{zone} VAV Box Component"},
                                            update_values={"Zone Minimum Air Flow Input Method": "Scheduled",
                                                        "Minimum Air Flow Fraction Schedule Name": f"{zone} VAV Customized Schedule"})
            elif self.environment == "B_SanFrancisco":
                model = Model(idf_file_name="./eplus_files/OfficeMedium_SF.idf",
                                weather_file="./eplus_files/USA_CA_San.Francisco.Intl.AP.724940_TMY3.epw",
                                eplus_naming_dict=eplus_extra_states,
                                tmp_idf_path=self.save_dir)
                for key, _ in eplus_extra_states.items():
                    model.add_configuration("Output:Variable",
                                            {"Key Value": key[1], "Variable Name": key[0], "Reporting Frequency": "Timestep"})
                
                # Setup controls to all VAV boxes
                self.control_zones = available_zones[3:]
                for zone in self.control_zones:
                    model.add_configuration("Schedule:Constant",
                                            {"Name": f"{zone} VAV Customized Schedule",
                                            "Schedule Type Limits Name": "Fraction",
                                            "Hourly Value": 0})
                    model.edit_configuration(idf_header_name="AirTerminal:SingleDuct:VAV:Reheat",
                                            identifier={"Name": f"{zone} VAV Box Component"},
                                            update_values={"Zone Minimum Air Flow Input Method": "Scheduled",
                                                        "Minimum Air Flow Fraction Schedule Name": f"{zone} VAV Customized Schedule"})
        
        elif self.environment == "C":
            available_zones = ["Amphitheater", "Lab", "Library",
                               "North-1", "North-2", "North-3", "North-G",
                               "South-1", "South-2", "South-3", "South-GF"]

            # Add state variables that we care about
            eplus_extra_states = {("Zone Air Relative Humidity", zone): f"{zone} humidity" for zone in available_zones}
            eplus_extra_states.update({("Zone Air System Sensible Heating Rate", f"{zone}"): f"{zone} vav heating energy" for zone in available_zones})
            eplus_extra_states.update({("Zone Air System Sensible Cooling Rate", f"{zone}"): f"{zone} vav cooling energy" for zone in available_zones})
            eplus_extra_states.update({("Zone Air Terminal VAV Damper Position", f"VAV HW Rht {zone}"): f"{zone} position" for zone in available_zones})
            eplus_extra_states[('Site Outdoor Air Drybulb Temperature', 'Environment')] = "outdoor temperature"
            eplus_extra_states[('Site Direct Solar Radiation Rate per Area', 'Environment')] = "site solar radiation"
            eplus_extra_states[('Facility Total HVAC Electric Demand Power', 'Whole Building')] = "total hvac"
            eplus_extra_states[('Schedule Value', 'HVACOperationSchd')] = "operations availability"


            model = Model(idf_file_name="./eplus_files/DOEE_V930.idf",
                        weather_file="./eplus_files/USA_CA_San.Francisco.Intl.AP.724940_TMY3.epw",
                        eplus_naming_dict=eplus_extra_states,
                        tmp_idf_path=self.save_dir)
            
            # Add them to the IDF file so we can retrieve them
            for key, _ in eplus_extra_states.items():
                model.add_configuration("Output:Variable",
                                        {"Key Value": key[1], "Variable Name": key[0], "Reporting Frequency": "Timestep"})
            
            # Setup controls to all VAV boxes
            self.control_zones = available_zones
            for zone in self.control_zones:
                model.add_configuration("Schedule:Constant",
                                        {"Name": f"{zone} VAV Customized Schedule",
                                        "Schedule Type Limits Name": "Fraction",
                                        "Hourly Value": 0.1})
                model.edit_configuration(idf_header_name="AirTerminal:SingleDuct:VAV:Reheat",
                                        identifier={"Name": f"VAV HW Rht {zone}"},
                                        update_values={"Zone Minimum Air Flow Input Method": "Scheduled",
                                                        "Constant Minimum Air Flow Fraction": "",
                                                        "Minimum Air Flow Fraction Schedule Name": f"{zone} VAV Customized Schedule"})
        else:
            raise ValueError("Environment not supported. Only B_Denver, B_SanFrancisco and C are supported.")
        self.model = model
    
    def set_start_date(self, run_duration=30, start_year=1994, start_month=1, start_day=1, timestep_per_hour=4):
        """Method to set the start date in the COBS environment model.

        Args:
            run_duration (int, optional): The number of days to simulate. Defaults to 30.
            start_year (int, optional): The year to start the simulation. Defaults to 1994.
            start_month (int, optional): The month to start the simulation. Defaults to 1.
            start_day (int, optional): The day to start the simulation. Defaults to 1.
            timestep_per_hour (int, optional): The number of timesteps per hour. Defaults to 4.
        """
        self.model.set_runperiod(days=run_duration, start_year=start_year, start_month=start_month, start_day=start_day, specify_year=True)
        self.model.set_timestep(timestep_per_hour=timestep_per_hour)

    def play(self, arm_idx, arm, eval_duration=30):
        """Method to simulate one arm pull

        Args:
            arm_idx (int): The index of the arm pulled
            arm (dict): Dictionary containing the polices for each group of zones
            eval_duration (int, optional): The simulation duration. Defaults to 30.

        Returns:
            tuple: A tuple containing the following values
                1. The normalized energy consumption. Normalized to make UCB converge faster
                2. The year at which the simulation took place
                3. The month at which the simulation took place
                4. The day at which the simulation took place
                5. The total energy consumption of the arm pull
        """
        # Uncomment the below code if each arm pull is supposed to simulate a random day
        # possible_years = [1994, 1997, 1998, 1999, 2002, 2003, 2004, 2005]
        # possible_months = range(12)
        # possible_dates = range(31)

        # year = random.choice(possible_years)
        # month = random.choice(possible_months) + 1

        # while True:
        #     day = random.choice(possible_dates) + 1
        #     try:
        #         date = datetime(year=year, month=month, day=day)
        #         if date.weekday() <= 4:
        #             # Date is a weekday
        #             break
        #     except ValueError:
        #         continue
        
        # Fixing the simulation day for all arm pulls (to make the problem a stationary one)
        year = 1991
        month = 1
        day = 1
        date = datetime(year=year, month=month, day=day)
        if not self.use_dummy_arms:
            self.set_start_date(run_duration=eval_duration, start_year=year, start_month=month, start_day=day)
            state = self.model.reset()
            total_energy = 0
            total_energy += state["total hvac"]
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
                        zone_action = 0.9/(1 + np.exp(-zone_action)) + 0.1
                        action.append({"priority": 0,
                                "component_type": "Schedule:Constant",
                                "control_type": "Schedule Value",
                                "actuator_key": f"{zone} VAV Customized Schedule",
                                "value": zone_action,
                                "start_time": state['timestep'] + 1})

                state = self.model.step(action)
                total_energy += state["total hvac"]
            if self.environment == "B_Denver":
                return -((total_energy - 16248682.536965799) / (144517646.5099022 - 16248682.536965799)), year, month, date.day, total_energy  # One month eval (30 days)
            elif self.environment == "B_SanFrancisco":
                return -((total_energy - 7253970.9438649295) / (127149241.43899378 - 7253970.9438649295)), year, month, date.day, total_energy  # One month eval (30 days)
            elif self.environment == "C":
                return -((total_energy - 17976704.46193411) / (57561696.237888955 - 17976704.46193411)), year, month, date.day, total_energy  # One month eval (30 days)

        elif self.use_dummy_arms:
            sigma = arm["sigma"]
            mu = arm["mu"]
            q = np.random.normal(mu, sigma)
            energy_mu = self.dummy_arms_energy[arm_idx]["mu"]
            energy_sigma = self.dummy_arms_energy[arm_idx]["sigma"]
            return q, year, month, date.day, np.random.normal(energy_mu, energy_sigma)

    def log_data(self, arm_name=None, arm_scores=None, arm_counts=None, flops=None,
                 initialize=False, arm_names=None, buffer_size=1,
                 start_year=None, start_month=None, start_day=None,
                 total_energy=None, arm_scores_all=None):
        """Method to log the data to a csv file

        Args:
            arm_name (str, optional): The name of the arm pulled. Defaults to None.
            arm_scores (float, optional): The UCB score of the arm selected. Defaults to None.
            arm_counts (int, optional): The number of times this arm has been pulled. Defaults to None.
            flops (float, optional): The computational complexity of pulling this arm (running time). Defaults to None.
            initialize (bool, optional): Boolean to indicate whether to write the headers of csv or not. Defaults to False.
            arm_names (list, optional): List containing the names of all arms. Defaults to None.
            buffer_size (int, optional): The buffer size to indicate the frequency at which Python actually writes to the file. Defaults to 1.
            start_year (int, optional): The year at which the arm pull took place. Defaults to None.
            start_month (int, optional): The month at which the arm pull tookplace. Defaults to None.
            start_day (int, optional): The day at which the arm pull took place. Defaults to None.
            total_energy (float, optional): The total energy consumption for the arm pull. Defaults to None.
            arm_scores_all (list, optional): The scores of all arms at this timestep. Defaults to None.
        """
        self.csv_loc = os.path.join(self.save_dir, "ucb_log_data.csv")
        self.csv_file_obj = open(self.csv_loc, "a+", buffering=buffer_size)
        if initialize:
            cols = ["datetime", "flops", "policy_name", "start_year",
                    "start_month", "start_day", "total_energy"]
            # self.csv_loc = os.path.join(self.save_dir, "ucb_log_data.csv")
            # self.csv_file_obj = open(self.csv_loc, "w+", buffering=buffer_size)
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
            with open(os.path.join(self.save_dir, "arm_scores_all.pkl"), "wb") as f:
                pickle.dump(arm_scores_all, f)


    def calc_ucb_value(self, policy_count, total_count, rho=2):
        """Method to calculate the UCB values for each arm

        Args:
            policy_count (list): List containing the number of times each arm has been pulled
            total_count (int): The total number of times all arms have been pulled

        Returns:
            list: List containing the UCB values for each arm
        """
        if total_count == 0 or policy_count == 0:
            return 0
        else:
            return np.sqrt(rho*np.log(total_count)/policy_count)


    def get_latest_arm_data(self, df):
        """Method to get the latest arm data from the read csv file
        
        Args:
            df (pandas dataframe): The dataframe containing the data from the csv file
        
        Returns:
            arm_scores (list): The scores of all arms
            arm_counts (list): The number of times each arm has been pulled
        """
        last_row = df.iloc[-1]
        arm_scores = []
        arm_counts = []
        for column in last_row.keys():
            if column.startswith("arm_"):
                if column.endswith("_score"):
                    arm_scores.append(last_row[column])
                elif column.endswith("_count"):
                    arm_counts.append(last_row[column])
        return arm_scores, arm_counts


    def run_ucb(self, epochs=None, arm_scores=None,
                arm_counts=None, arm_scores_all=None,
                rho=2, eval_duration=30):
        """Method to run the UCB algorithm

        Args:
            epochs (int, optional): The number of epochs to run the algorithm for. Defaults to None (infinite).
            arm_scores (list, optional): The scores of all arms. Defaults to None. Only if continuing from a previous run.
            arm_counts (list, optional): The number of times each arm has been pulled. Defaults to None. Only if continuing from a previous run.
            arm_scores_all (list, optional): The scores of all arms at each timestep. Defaults to None. Only if continuing from a previous run.
            rho (int, optional): The exploration parameter. Defaults to 2.
            eval_duration (int, optional): The duration for which the simulation should run. Defaults to 30.

        """
        self.make_log_dir()
        if self.pickup_from is None:
            self.log_data(initialize=True)
        else:
            df = pd.read_csv(os.path.join(self.save_dir, "ucb_log_data.csv"))
            arm_scores, arm_counts = self.get_latest_arm_data(df)
            arm_scores_all = pickle.load(open(os.path.join(self.save_dir, "arm_scores_all.pkl"), "rb"))
        if not self.use_dummy_arms:
            self.make_cobs_model()
        if arm_scores is None:
            arm_scores = np.ones((len(self.arms))) * np.inf

        if arm_scores_all is None:
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
                if self.use_dummy_arms:
                    arm_name = list(self.dummy_arms.keys())[chosen_arm_idx]
                    q_policy, start_year, start_month, start_day, total_energy = self.play(chosen_arm_idx, self.dummy_arms[chosen_arm_idx], eval_duration=eval_duration)
                else:
                    arm_name = list(self.arms.keys())[chosen_arm_idx]
                    q_policy, start_year, start_month, start_day, total_energy = self.play(chosen_arm_idx, self.arms[chosen_arm_idx], eval_duration=eval_duration)

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
                              total_energy=total_energy, arm_scores_all=arm_scores_all)
                if epochs is not None:
                    if epoch == epochs:
                        self.csv_file_obj.close()
                        break
        finally:
            traceback.print_exc()
            self.csv_file_obj.close()
