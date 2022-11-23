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
        '--seed',
        help='Number of episodes to run',
        type=int,
        default=1911,
    )
    parser.add_argument(
        '--diverse',
        help='1 (True) or 0 (False), consider policy library with diverse policies',
        type=int,
        default=1,
    )
    parser.add_argument(
        '--all',
        help='1 (True) or 0 (False), consider policy library cross seeds or not',
        type=int,
        default=1,
    )
    parser.add_argument(
        '--clusters',
        help='number indicating the number of policy clusters',
        type=int,
        default=-1,
    )
    parser.add_argument(
        '--episodes',
        help='Set to an integer for number of episodes',
        type=int,
        default=1
    )
    parser.add_argument(
        '--from_scratch',
        help='train from scratch or use existing policy',
        type=int,
        default=0
    )
    parser.add_argument(
        '--continue_train',
        help='continue previous training to desired episodes',
        type=int,
        default=0
    )
    parser.add_argument(
        '--single_agent',
        help='using single-agent to control all setpoints or not',
        type=int,
        default=0
    )
    parser.add_argument(
        '--flexlab',
        help='using flexlab IDF or not',
        type=int,
        default=0
    )
    parser.add_argument(
        '--no_blinds',
        help='Shall we consider the policy trained with extra blinds',
        type=int,
        default=0
    )
    parser.add_argument(
        '--patience',
        help='How many episodes to wait until reverse to the best policy',
        type=int,
        default=5
    )
    parser.add_argument(
        '--ignore_zero',
        help='Should we train when the system is off',
        type=int,
        default=0
    )
    parser.add_argument(
        '--change_rotation_location',
        help='1 for california and 45 deg rotation',
        type=int,
        default=0
    )
    parser.add_argument(
        '--doee',
        help='using doee IDF or not',
        type=int,
        default=0
    )
    
    args = parser.parse_args()
    np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)
    
    print("============================================================================================")
    # set device to cpu or cuda
    device = torch.device('cpu')

    if args.flexlab:
        available_zones = ["FlexLab-X3-ZoneA", "FlexLab-X3-ZoneB"]
        airloops = {'FlexLab-X3-ZoneA': "Sys-A",
                    'FlexLab-X3-ZoneB': "Sys-B"}

        # Add state variables that we care about
        eplus_extra_states = {("Zone Air Relative Humidity", zone): f"{zone} humidity" for zone in available_zones}
        eplus_extra_states.update({("Air System Electric Energy", f"{airloops[zone]}"): f"{zone} vav energy" for zone in available_zones})  # Could be Power
        eplus_extra_states.update({("Zone Air Terminal VAV Damper Position", f"{zone} Direct Air"): f"{zone} vav pos" for zone in available_zones})  # Could be Power
        eplus_extra_states.update({("Zone Air Terminal Outdoor Air Volume Flow Rate", f"{zone} Direct Air"): f"{zone} vav flow" for zone in available_zones})  # Could be Power
        eplus_extra_states[('Site Outdoor Air Drybulb Temperature', 'Environment')] = "outdoor temperature"
        eplus_extra_states[('Site Direct Solar Radiation Rate per Area', 'Environment')] = "site solar radiation"
        eplus_extra_states[('Facility Total Electric Demand Power', 'Whole Building')] = "total hvac"
    elif args.doee:
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
    else:
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

    choose_an_initial_option = "Scratch"
    if not args.from_scratch:
        agent_results = agent_result_all[f'{"diverse" if args.diverse else "optimal_only"}_{"all" if args.all else "seed"}']
        if args.clusters != -1:
            if args.doee:
                agent_results = result_doee[args.clusters]
            elif args.change_rotation_location:
                agent_results = result_rotate[args.clusters]
            else:
                agent_results = result[args.clusters]
        choose_an_initial_option = sorted(list(agent_results.keys()))[args.seed % len(agent_results)]
        agent_results = agent_results[choose_an_initial_option]

    checkpoint_path = f"policy_library/transfer/new/{choose_an_initial_option}_seed_{args.seed}_diverse_{args.diverse}_all_{args.all}_scratch_{args.from_scratch}_cluster_{args.clusters}_SA_{args.single_agent}_ignore_{args.ignore_zero}_newloc_{args.change_rotation_location}"
    checkpoint_path += f"_no_blinds" if args.no_blinds else ""
    checkpoint_path += f"_flexlab" if args.flexlab else ""
    checkpoint_path += f"_doee" if args.doee else ""
    os.makedirs(f"{checkpoint_path}", exist_ok=True)
    if args.change_rotation_location:
        idf_file = "./eplus_files/OfficeMedium_SF.idf"
        weather_file = "./eplus_files/USA_CA_San.Francisco.Intl.AP.724940_TMY3.epw"
    elif args.flexlab:
        idf_file = "./eplus_files/HVAC_Sha_csv_AB.idf"
        weather_file = "./eplus_files/USA_CO_Denver-Aurora-Buckley.AFB_.724695_TMY3.epw"
    elif args.doee:
        idf_file = "./eplus_files/DOEE_V930.idf"
        weather_file = "./eplus_files/USA_CA_San.Francisco.Intl.AP.724940_TMY3.epw"
    else:
        idf_file = "./eplus_files/OfficeMedium_Denver.idf"
        weather_file = "./eplus_files/USA_CO_Denver-Aurora-Buckley.AFB_.724695_TMY3.epw"
    model = Model(idf_file_name=idf_file,
                  weather_file=weather_file,
                  eplus_naming_dict=eplus_extra_states,
                  tmp_idf_path=checkpoint_path)
    
    initial_episode = 0
    log_mode = "w"
    if args.continue_train:
        with open(f"{checkpoint_path}-log.csv", 'r') as previous_log:
            initial_episode = max(len(previous_log.readlines()) - 2, 0)
        log_mode = "a"
    log_f = open(f"{checkpoint_path}-log.csv", log_mode)
    
    # Add them to the IDF file so we can retrieve them
    for key, _ in eplus_extra_states.items():
        model.add_configuration("Output:Variable",
                                {"Key Value": key[1], "Variable Name": key[0], "Reporting Frequency": "Timestep"})
    
    # Setup controls to all VAV boxes
    if args.flexlab:
        control_zones = available_zones
    elif args.doee:
        control_zones = available_zones
    else:
        control_zones = available_zones[3:]
    for zone in control_zones:
        model.add_configuration("Schedule:Constant",
                                {"Name": f"{zone} VAV Customized Schedule",
                                 "Schedule Type Limits Name": "Fraction",
                                 "Hourly Value": 0})
        if args.flexlab:
            header_name = "AirTerminal:SingleDuct:VAV:NoReheat"
            vav_identifier = f"{zone} Direct Air"
        elif args.doee:
            header_name = "AirTerminal:SingleDuct:VAV:Reheat"
            vav_identifier = f"VAV HW Rht {zone}"
        else:
            header_name = "AirTerminal:SingleDuct:VAV:Reheat"
            vav_identifier = f"{zone} VAV Box Component"
        model.edit_configuration(idf_header_name=header_name,
                                 identifier={"Name": vav_identifier},
                                 update_values={"Zone Minimum Air Flow Input Method": "Scheduled",
                                                "Constant Minimum Air Flow Fraction": "",
                                                "Minimum Air Flow Fraction Schedule Name": f"{zone} VAV Customized Schedule"})

    # Environment setup
    model.set_runperiod(*(30, 2000, 1, 1))
    model.set_timestep(4)

    if not args.continue_train:
        log_f.write(f'initial: {choose_an_initial_option} - {"diverse" if args.diverse else "optimal_only"}{"_no_blind_" if args.no_blinds else "_"}{"all" if args.all else "seed"}{args.clusters}\n')
        log_f.flush()

    selected_agents = list()
    if not args.single_agent:
        for i, zone in enumerate(control_zones):
            agent = PPO(1 + 1 + 1 + 1 + 1 + 1, # State dimension, own temperature + humidity + outdoor temp + solar + occupancy + hour
                        1, # Action dimension, 1 for each zone
                        0.003, 0.0005, 1, 10, 0.2, has_continuous_action_space=True, action_std_init=0.6, 
                        device=device,
                        diverse_policies=list(), diverse_weight=0, diverse_increase=True)
            if args.continue_train:
                agent.load(f"{checkpoint_path}/agent_{i}.pth")
            elif not args.from_scratch:
                agent.load(agent_results[zone])
            selected_agents.append(agent)
    else:
        agent = PPO(len(control_zones) * 3 + 3,
                    len(control_zones),
                    0.003, 0.0005, 1, 10, 0.2, has_continuous_action_space=True, action_std_init=0.6, 
                    device=device,
                    diverse_policies=list(), diverse_weight=0, diverse_increase=True)
        if args.continue_train:
            agent.load(f"{checkpoint_path}/agent.pth")
    
    best_reward = -1
    patience = args.patience
    
    for ep in range(initial_episode, args.episodes):
        state = model.reset()
        if args.doee:
            for zone in control_zones:
                state[f"{zone} vav energy"] = state[f"{zone} vav heating energy"] + state[f"{zone} vav cooling energy"]
        total_energy = state["total hvac"]
        
        while not model.is_terminate():
            for zone in state["occupancy"]:
                state["occupancy"][zone] = 1 if state["occupancy"][zone] > 0 else 0

            # Transfer the state into the format of only selected states
            agent_state = [state["outdoor temperature"], state["site solar radiation"], state["time"].hour]
                
            action = list()
            for i, zone in enumerate(control_zones):
                if selected_agents:
                    action.append(selected_agents[i].select_action(agent_state + [state[f"{zone} humidity"], state["temperature"][zone], state["occupancy"][zone]]))
                else:
                    # Single agent case
                    agent_state.append(state[f"{zone} humidity"])
                    agent_state.append(state["temperature"][zone])
                    agent_state.append(state["occupancy"][zone])
            
            if not selected_agents:
                action = agent.select_action(agent_state)

            action = np.array(action)
            action = list(0.9/(1 + np.exp(-action)) + 0.1)
            
            actions = list()
            for i, zone in enumerate(control_zones):
                actions.append({"priority": 0,
                                "component_type": "Schedule:Constant",
                                "control_type": "Schedule Value",
                                "actuator_key": f"{zone} VAV Customized Schedule",
                                "value": action[i],
                                "start_time": state['timestep'] + 1})
            state = model.step(actions)
            if args.doee:
                for zone in control_zones:
                    state[f"{zone} vav energy"] = state[f"{zone} vav heating energy"] + state[f"{zone} vav cooling energy"]
            
            if selected_agents:
                for i, zone in enumerate(control_zones):
                    selected_agents[i].buffer.rewards.append(-state[f"{zone} vav energy"])  # -state[f"{airloops[zone]} energy"]
                    selected_agents[i].buffer.is_terminals.append(state["terminate"])

                    if args.ignore_zero and -state[f"{zone} vav energy"] <= 10:
                        selected_agents[i].buffer.remove_last()
            else:
                agent.buffer.rewards.append(state["total hvac"])
                agent.buffer.is_terminals.append(state["terminate"])

                if args.ignore_zero and -state[f"total hvac"] <= 10:
                    agent.buffer.remove_last()

            total_energy += state["total hvac"]
        
        if selected_agents:
            for i in range(len(selected_agents)):
                if ((ep + 1) % 200) == 0:
                    selected_agents[i].decay_action_std(0.02, 0.1)
                selected_agents[i].update()
                if best_reward == -1 or int(best_reward) >= int(total_energy):
                    selected_agents[i].save(f"{checkpoint_path}/agent_{i}.pth")
                    best_reward = total_energy
                else:
                    patience -= 1
                    if patience == 0:
                        selected_agents[i].load(f"{checkpoint_path}/agent_{i}.pth")
                        patience = args.patience
        else:
            if ((ep + 1) % 200) == 0:
                agent.decay_action_std(0.02, 0.1)
            agent.update()
            if best_reward == -1 or int(best_reward) >= int(total_energy):
                agent.save(f"{checkpoint_path}/agent.pth")
                best_reward = total_energy
            else:
                patience -= 1
                if patience == 0:
                    agent.load(f"{checkpoint_path}/agent.pth")
                    patience = args.patience

        print(f"[{datetime.datetime.now()}]Total energy: {total_energy}")
        log_f.write(f"[{datetime.datetime.now()}]Total energy: {total_energy}\n")
        log_f.flush()

    log_f.close()
    print("Done")