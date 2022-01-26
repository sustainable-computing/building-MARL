from email.policy import default
import sys
import os
from ppo import PPO
import argparse
import torch
import numpy as np
import glob
import datetime
sys.path.insert(0, "/home/tianyu/building-MARL/")
from cobs import Model
Model.set_energyplus_folder("/usr/local/EnergyPlus-9-3-0/")


if __name__ == '__main__':
    # Setup run parameters
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--gpu',
        help='Which GPU to train on',
        type=int,
        default=3,
    )
    parser.add_argument(
        '--seed',
        help='Number of episodes to run',
        type=int,
        default=1911,
    )
    parser.add_argument(
        '--train_on',
        help='Set to -1 for parallel training, 0-14 otherwise indicating which zone to train, other zones will use rule-based controller',
        type=int,
        default=11
    )
    parser.add_argument(
        '--diverse_training',
        help='Set to an integer for diverse training, where all other agents are previous policies. 0 to turn off',
        type=int,
        default=1
    )
    parser.add_argument(
        '--keeps_training',
        help='Set to 1 to keeps training. 0 to turn off',
        type=int,
        default=0
    )
    
    
    args = parser.parse_args()
    
    print("============================================================================================")
    # set device to cpu or cuda
    device = torch.device('cpu')

    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        torch.cuda.empty_cache()
        print(f"Device set to: {torch.cuda.get_device_name(device)}")
    else:
        print("Device set to: cpu")
    print("============================================================================================")
    run_num = 19
    
    checkdir_dir = f"PPO_weights/PPO_{args.seed}_{run_num}_single_env_training"
    
    log_f = open(f"{checkdir_dir}/log_evaluate_policies_{args.diverse_training}", "w+")

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


    os.makedirs(f"{checkdir_dir}/evaluation", exist_ok=True)
    model = Model(idf_file_name="./eplus_files/OfficeMedium_Denver.idf",
                  weather_file="./eplus_files/USA_CO_Denver-Aurora-Buckley.AFB_.724695_TMY3.epw",
                  eplus_naming_dict=eplus_extra_states,
                  tmp_idf_path=f"{checkdir_dir}/evaluation")
    
    # Add them to the IDF file so we can retrieve them
    for key, _ in eplus_extra_states.items():
        model.add_configuration("Output:Variable",
                                {"Key Value": key[1], "Variable Name": key[0], "Reporting Frequency": "Timestep"})
    
    # Setup controls to all VAV boxes
    control_zones = available_zones[4:5] + available_zones[14:]
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
    model.set_runperiod(*(30, 1991, 7, 1))
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
    agent_save_paths = list(glob.glob(f"PPO_weights/PPO_{args.seed}_{run_num}_single_env_training/agent_*.pth")) + \
        list(glob.glob(f"PPO_weights/PPO_{args.seed}_{run_num}_single_env_training/*/agent_{args.diverse_training}.pth"))
    
    agent_results = dict()
    
    # for test_zone in control_zones:
    #     for agent_path in agent_save_paths:
    #         agent.load(agent_path)

    #         state = model.reset()
    #         total_energy = state["total hvac"]
            
    #         while not model.is_terminate():
    #             # Transfer the state into the format of only selected states
    #             action = agent.select_action([state["outdoor temperature"],
    #                                           state["site solar radiation"],
    #                                           state["time"].hour,
    #                                           state[f"{test_zone} humidity"],
    #                                           state["temperature"][test_zone],
    #                                           state["occupancy"][test_zone]])

    #             action = np.array(action)
    #             action = 1/(1 + np.exp(-action))
                
    #             actions = list()
    #             actions.append({"priority": 0,
    #                             "component_type": "Schedule:Constant",
    #                             "control_type": "Schedule Value",
    #                             "actuator_key": f"{test_zone} VAV Customized Schedule",
    #                             "value": action,
    #                             "start_time": state['timestep'] + 1})
    #             state = model.step(actions)
                
    #             total_energy += state["total hvac"]
            
    #         agent.buffer.clear()

    #         print(f"[{datetime.datetime.now()}]Test agent: {agent_path}\tTest Zone: {test_zone}\tTotal energy: {total_energy}")
    #         log_f.write(f"[{datetime.datetime.now()}]Test agent: {agent_path}\tTest Zone: {test_zone}\tTotal energy: {total_energy}\n")
    #         log_f.flush()
            
    #         if test_zone not in agent_results:
    #             agent_results[test_zone] = dict()
    #         agent_results[test_zone][int(total_energy)] = agent_path
    agent_results = {"Core_mid": {1: "PPO_weights/PPO_1911_19_single_env_training/1.0/agent_1.pth"},
                     "Perimeter_mid_ZN_3": {1: "PPO_weights/PPO_1911_19_single_env_training/0.01/agent_1.pth"},
                     "Perimeter_mid_ZN_2": {1: "PPO_weights/PPO_1911_19_single_env_training/1.0/agent_1.pth"},
                     "Perimeter_mid_ZN_1": {1: "PPO_weights/PPO_1911_19_single_env_training/1.0/agent_1.pth"},
                     "Perimeter_mid_ZN_4": {1: "PPO_weights/PPO_1911_19_single_env_training/1.0/agent_1.pth"}
    }
    
    checkpoint_path = f"{checkdir_dir}/further_2"
    os.makedirs(checkpoint_path, exist_ok=True)

    selected_agents = list()
    for zone in control_zones:
        agent = PPO(1 + 1 + 1 + 1 + 1 + 1, # State dimension, own temperature + humidity + outdoor temp + solar + occupancy + hour
                    1, # Action dimension, 1 for each zone
                    0.003, 0.0005, 1, 10, 0.2, has_continuous_action_space=True, action_std_init=0.6, 
                    device=device,
                    diverse_policies=list(), diverse_weight=0, diverse_increase=True)
        agent.load(agent_results[zone][min(agent_results[zone].keys())])
        selected_agents.append(agent)
    
    num_eps = 1
    if args.keeps_training:
        num_eps = 5000
        log_f.close()
        log_f = open(f"{checkpoint_path}/log_evaluate_policies_{args.diverse_training}", "w+")
    
    for ep in range(num_eps):
        state = model.reset()
        total_energy = state["total hvac"]
        
        while not model.is_terminate():
            # Transfer the state into the format of only selected states
            agent_state = [state["outdoor temperature"], state["site solar radiation"], state["time"].hour]
            
            action = list()
            for i, zone in enumerate(control_zones):
                action.append(selected_agents[i].select_action(agent_state + [state[f"{zone} humidity"], state["temperature"][zone], state["occupancy"][zone]]))
                selected_agents[i].buffer.rewards.append(-state[f"{zone} vav energy"])  # -state[f"{airloops[zone]} energy"]
                selected_agents[i].buffer.is_terminals.append(state["terminate"])

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
            
            total_energy += state["total hvac"]
        
        for i in range(len(selected_agents)):
            if args.keeps_training:
                selected_agents[i].update()
                selected_agents[i].save(f"{checkpoint_path}/agent_{i}.pth")
            else:
                selected_agents[i].buffer.clear()

        print(f"[{datetime.datetime.now()}]Total energy: {total_energy}")
        log_f.write(f"[{datetime.datetime.now()}]Total energy: {total_energy}\n")
        log_f.flush()

    log_f.close()
    print("Done")