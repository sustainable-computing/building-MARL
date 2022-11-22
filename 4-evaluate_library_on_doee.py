from ast import arg
from local_setting import *
from email.policy import default
import sys
import os
from ppo import PPO
import argparse
import torch
import numpy as np
import glob
import datetime
sys.path.insert(0, file_path)
from cobs import Model
Model.set_energyplus_folder(energyplus_location)


if __name__ == '__main__':
    # Setup run parameters
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--evaluate_on',
        help='0-1 indicating which zone to train, other zones will use rule-based controller',
        type=int,
        default=11
    )
    parser.add_argument(
        '--evaluate_start',
        help='Which weight to start with (index)',
        type=int,
        default=0
    )
    parser.add_argument(
        '--evaluate_length',
        help='how many to evaluate (index)',
        type=int,
        default=-1
    )
    
    args = parser.parse_args()
    
    print("============================================================================================")
    # set device to cpu or cuda
    device = torch.device('cpu')

    available_zones = ["Amphitheater", "Lab", "Library",
                       "North-1", "North-2", "North-3", "North-G",
                       "South-1", "South-2", "South-3", "South-GF"]

    # Add state variables that we care about
    eplus_extra_states = {("Zone Air Relative Humidity", zone): f"{zone} humidity" for zone in available_zones}
    eplus_extra_states.update({("Zone Air System Sensible Heating Rate", f"{zone}"): f"{zone} vav heating energy" for zone in available_zones})
    eplus_extra_states.update({("Zone Air System Sensible Cooling Rate", f"{zone}"): f"{zone} vav cooling energy" for zone in available_zones})
    eplus_extra_states.update({("Zone Air Terminal VAV Damper Position", f"VAV HW Rht {zone}"): f"{zone} position" for zone in available_zones})
    # eplus_extra_states.update({("Air System Electric Energy", airloop): f"{airloop} energy" for airloop in set(airloops.values())})
    eplus_extra_states[('Site Outdoor Air Drybulb Temperature', 'Environment')] = "outdoor temperature"
    eplus_extra_states[('Site Direct Solar Radiation Rate per Area', 'Environment')] = "site solar radiation"
    eplus_extra_states[('Facility Total HVAC Electric Demand Power', 'Whole Building')] = "total hvac"
    eplus_extra_states[('Schedule Value', 'HVACOperationSchd')] = "operations availability"


    os.makedirs(f"policy_library/evaluation/doee-{args.evaluate_on}-{args.evaluate_start}-{args.evaluate_length}", exist_ok=True)
    model = Model(idf_file_name="./eplus_files/DOEE_V930.idf",
                  weather_file="./eplus_files/USA_CA_San.Francisco.Intl.AP.724940_TMY3.epw",
                  eplus_naming_dict=eplus_extra_states,
                  tmp_idf_path=f"policy_library/evaluation/doee-{args.evaluate_on}-{args.evaluate_start}-{args.evaluate_length}")
    log_f = open(f"policy_library/evaluation/doee-{args.evaluate_on}-{args.evaluate_start}-{args.evaluate_length}-log.csv", "w+")
    
    # Add them to the IDF file so we can retrieve them
    for key, _ in eplus_extra_states.items():
        model.add_configuration("Output:Variable",
                                {"Key Value": key[1], "Variable Name": key[0], "Reporting Frequency": "Timestep"})
    
    # Setup controls to all VAV boxes
    control_zones = [available_zones[args.evaluate_on]]
    for zone in control_zones:
        model.add_configuration("Schedule:Constant",
                                {"Name": f"{zone} VAV Customized Schedule",
                                 "Schedule Type Limits Name": "Fraction",
                                 "Hourly Value": 0})
        model.edit_configuration(idf_header_name="AirTerminal:SingleDuct:VAV:Reheat",
                                 identifier={"Name": f"VAV HW Rht {zone}"},
                                 update_values={"Zone Minimum Air Flow Input Method": "Scheduled",
                                                "Constant Minimum Air Flow Fraction": "",
                                                "Minimum Air Flow Fraction Schedule Name": f"{zone} VAV Customized Schedule"})

    # Environment setup
    model.set_runperiod(*(7, 1990, 1, 1))
    model.set_timestep(4)
    
    # Agent setup
    # num_rl_agent = 15
    # existing_policies = list()
    # if args.train_on == -1 and args.diverse_training != 0:
    #     existing_policies = list(glob.glob(f"PPO_weights/PPO_{args.seed}_{run_num}_single_env_training/agent_*.pth"))
    agent = PPO(1 + 1 + 1 + 1 + 1 + 1, # State dimension, own temperature + humidity + outdoor temp + solar + occupancy + hour
                1, # Action dimension, 1 for each zone
                0.003, 0.0005, 1, 10, 0.2, has_continuous_action_space=True, action_std_init=0.2, 
                device=device,
                diverse_policies=list(), diverse_weight=0, diverse_increase=True)
    agent_save_paths = sorted(list(glob.glob(f"policy_library/**.pth")))[args.evaluate_start:args.evaluate_start + args.evaluate_length]
    test_zone = control_zones[0]

    print(agent_save_paths)

    for agent_path in agent_save_paths:
        agent.load(agent_path)

        state = model.reset()
        for zone in control_zones:
            state[f"{zone} vav energy"] = state[f"{zone} vav heating energy"] + state[f"{zone} vav cooling energy"]
        total_energy = state["total hvac"]
        
        while not model.is_terminate():
            occupancy = 1 if state["occupancy"][test_zone] > 0 else 0
            # Transfer the state into the format of only selected states
            action = agent.select_action([state["outdoor temperature"],
                                          state["site solar radiation"],
                                          state["time"].hour,
                                          state[f"{test_zone} humidity"],
                                          state["temperature"][test_zone],
                                          occupancy])

            action = np.array(action)
            action = 0.9/(1 + np.exp(-action)) + 0.1
            # print(action)
            
            actions = list()
            actions.append({"priority": 0,
                            "component_type": "Schedule:Constant",
                            "control_type": "Schedule Value",
                            "actuator_key": f"{test_zone} VAV Customized Schedule",
                            "value": action,
                            "start_time": state['timestep'] + 1})
            state = model.step(actions)
            for zone in control_zones:
                state[f"{zone} vav energy"] = state[f"{zone} vav heating energy"] + state[f"{zone} vav cooling energy"]
            
            total_energy += state["total hvac"]
        
        agent.buffer.clear()

        print(f"[{datetime.datetime.now()}]Test agent: {agent_path}\tTest Zone: {test_zone}\tTotal energy: {total_energy}")
        log_f.write(f"{datetime.datetime.now()},{agent_path},{test_zone},{total_energy}\n")
        log_f.flush()

    log_f.close()
    print("Done")