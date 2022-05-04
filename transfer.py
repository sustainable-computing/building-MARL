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
        '--no_blinds',
        help='Shall we consider the policy trained with extra blinds',
        type=int,
        default=0
    )
    
    args = parser.parse_args()
    np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)
    
    print("============================================================================================")
    # set device to cpu or cuda
    device = torch.device('cpu')

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

    checkpoint_path = f"policy_library/transfer/new/seed_{args.seed}_diverse_{args.diverse}_all_{args.all}_scratch_{args.from_scratch}"
    checkpoint_path += f"_no_blinds" if args.no_blinds else ""
    os.makedirs(f"{checkpoint_path}", exist_ok=True)
    model = Model(idf_file_name="./eplus_files/OfficeMedium_Denver.idf",
                  weather_file="./eplus_files/USA_CO_Denver-Aurora-Buckley.AFB_.724695_TMY3.epw",
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

    # Environment setup
    model.set_runperiod(*(30, 1991, 1, 1))
    model.set_timestep(4)

    agent_results = agent_result_all[f'{"diverse" if args.diverse else "optimal_only"}_{"all" if args.all else "seed"}']
    choose_an_initial_option = sorted(list(agent_results.keys()))[args.seed % len(agent_results)]
    agent_results = agent_results[choose_an_initial_option]

    if not args.continue_train:
        log_f.write(f'initial: {choose_an_initial_option} - {"diverse" if args.diverse else "optimal_only"}{"_no_blind_" if args.no_blinds else "_"}{"all" if args.all else "seed"}\n')
        log_f.flush()


    selected_agents = list()
    for i, zone in enumerate(control_zones):
        agent = PPO(1 + 1 + 1 + 1 + 1 + 1, # State dimension, own temperature + humidity + outdoor temp + solar + occupancy + hour
                    1, # Action dimension, 1 for each zone
                    0.003, 0.0005, 1, 10, 0.2, has_continuous_action_space=True, action_std_init=0.4, 
                    device=device,
                    diverse_policies=list(), diverse_weight=0, diverse_increase=True)
        if args.continue_train:
            agent.load(f"{checkpoint_path}/agent_{i}.pth")
        elif not args.from_scratch:
            agent.load(agent_results[zone])
        selected_agents.append(agent)
    
    best_reward = -1
    
    for ep in range(initial_episode, args.episodes):
        state = model.reset()
        total_energy = state["total hvac"]
        
        while not model.is_terminate():
            for zone in state["occupancy"]:
                state["occupancy"][zone] = 1 if state["occupancy"][zone] > 0 else 0

            # Transfer the state into the format of only selected states
            agent_state = [state["outdoor temperature"], state["site solar radiation"], state["time"].hour]
            
            action = list()
            for i, zone in enumerate(control_zones):
                action.append(selected_agents[i].select_action(agent_state + [state[f"{zone} humidity"], state["temperature"][zone], state["occupancy"][zone]]))

            action = np.array(action)
            action = list(1/(1 + np.exp(-action)))
            
            actions = list()
            for i, zone in enumerate(control_zones):
                actions.append({"priority": 0,
                                "component_type": "Schedule:Constant",
                                "control_type": "Schedule Value",
                                "actuator_key": f"{zone} VAV Customized Schedule",
                                "value": action[i],
                                "start_time": state['timestep'] + 1})
            state = model.step(actions)
            
            for i, zone in enumerate(control_zones):
                selected_agents[i].buffer.rewards.append(-state[f"{zone} vav energy"])  # -state[f"{airloops[zone]} energy"]
                selected_agents[i].buffer.is_terminals.append(state["terminate"])

            total_energy += state["total hvac"]

            # for i, zone in enumerate(control_zones):
            #     if -state[f"{zone} vav energy"] == 0:
            #         selected_agents[i].buffer.remove_last()
        
        for i in range(len(selected_agents)):
            if ((ep + 1) % 100) == 0:
                selected_agents[i].decay_action_std(0.02, 0.1)
            selected_agents[i].update()
            if best_reward == -1 or int(best_reward) >= int(total_energy):
                selected_agents[i].save(f"{checkpoint_path}/agent_{i}.pth")
                best_reward = total_energy
            else:
                selected_agents[i].load(f"{checkpoint_path}/agent_{i}.pth")

        print(f"[{datetime.datetime.now()}]Total energy: {total_energy}")
        log_f.write(f"[{datetime.datetime.now()}]Total energy: {total_energy}\n")
        log_f.flush()

    log_f.close()
    print("Done")