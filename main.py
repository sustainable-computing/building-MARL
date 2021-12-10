import sys
import os
from ppo import PPO
import argparse
import torch
import numpy as np
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
        '--lr_actor',
        help='Actor net learn rate',
        type=float,
        default=0.0003,
    )
    parser.add_argument(
        '--lr_critic',
        help='Critic net learn rate',
        type=float,
        default=0.001,
    )
    parser.add_argument(
        '--gamma',
        help='Discount factor',
        type=float,
        default=1,
    )
    parser.add_argument(
        '--k_epochs',
        help='Update policy for K epochs',
        type=int,
        default=10,
    )
    parser.add_argument(
        '--eps_clip',
        help='Clip parameter for PPO',
        type=float,
        default=0.2,
    )
    parser.add_argument(
        '--episodes',
        help='Number of episodes to run',
        type=int,
        default=5000,
    )
    parser.add_argument(
        '--seed',
        help='Number of episodes to run',
        type=int,
        default=19,
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
    run_num_pretrained = 0
    checkpoint_path = f"PPO_weights/PPO_{args.seed}_{run_num_pretrained}.pth"
    os.makedirs(f"PPO_weights", exist_ok=True)
    log_f = open("log", "w+")


    available_zones = ['TopFloor_Plenum', 'MidFloor_Plenum', 'FirstFloor_Plenum',
                       'Core_top', 'Core_mid', 'Core_bottom',
                       'Perimeter_top_ZN_3', 'Perimeter_top_ZN_2', 'Perimeter_top_ZN_1', 'Perimeter_top_ZN_4',
                       'Perimeter_bot_ZN_3', 'Perimeter_bot_ZN_2', 'Perimeter_bot_ZN_1', 'Perimeter_bot_ZN_4',
                       'Perimeter_mid_ZN_3', 'Perimeter_mid_ZN_2', 'Perimeter_mid_ZN_1', 'Perimeter_mid_ZN_4']

    # Add state variables that we care about
    eplus_extra_states = {("Zone Air Relative Humidity", zone): f"{zone} humidity" for zone in available_zones}
    eplus_extra_states[('Site Outdoor Air Drybulb Temperature', 'Environment')] = "outdoor temperature"
    eplus_extra_states[('Site Direct Solar Radiation Rate per Area', 'Environment')] = "site solar radiation"

    model = Model(idf_file_name="./eplus_files/OfficeMedium_Denver.idf",
                  weather_file="./eplus_files/USA_CO_Denver-Aurora-Buckley.AFB_.724695_TMY3.epw",
                  eplus_naming_dict=eplus_extra_states)
    
    # Add them to the IDF file so we can retrieve them
    for key, _ in eplus_extra_states.items():
        model.add_configuration("Output:Variable",
                                {"Key Value": key[1], "Variable Name": key[0], "Reporting Frequency": "Timestep"})
    
    # Setup controls to all VAV boxes
    for zone in available_zones:
        if "Plenum" not in zone:
            model.add_configuration("Schedule:Constant",
                                    {"Name": f"{zone} VAV Customized Schedule",
                                     "Schedule Type Limits Name": "Fraction",
                                     "Hourly Value": 0})
            model.edit_configuration(idf_header_name="AirTerminal:SingleDuct:VAV:Reheat",
                                     identifier={"Name": f"{zone} VAV Box Component"},
                                     update_values={"Zone Minimum Air Flow Input Method": "Scheduled",
                                                    "Minimum Air Flow Fraction Schedule Name": f"{zone} VAV Customized Schedule"})

    # Environment setup
    model.set_runperiod(*(10, 1991, 7, 1))
    model.set_timestep(4)
    
    # Agent setup
    agent = PPO(15 + 15 + 1 + 1 + 15 + 1, # State dimension, temp + humidity + outdoor temp + solar + occupancy + hour
                15, # Action dimension, 1 for each zone
                args.lr_actor, args.lr_critic, args.gamma, args.k_epochs, args.eps_clip, has_continuous_action_space=True, action_std_init=0.6, 
                device=device,
                diverse_policies=list(), diverse_weight=0)
    
    for ep in range(args.episodes):
        state = model.reset()
        total_energy = state["energy"]
        while not model.is_terminate():
            # Transfer the state into the format of only selected states
            agent_state = [state["outdoor temperature"], state["site solar radiation"], state["time"].hour]
            for zone in available_zones:
                if "Plenum" not in zone:
                    agent_state.append(state[f"{zone} humidity"])
                    agent_state.append(state["temperature"][zone])
                    agent_state.append(state["occupancy"][zone])
            
            # Get action and round to 0~1
            action = agent.select_action(agent_state)
            action = list(1/(1 + np.exp(-action)))
            
            actions = list()
            for zone in available_zones:
                if "Plenum" not in zone:
                    actions.append({"priority": 0,
                                    "component_type": "Schedule:Constant",
                                    "control_type": "Schedule Value",
                                    "actuator_key": f"{zone} VAV Customized Schedule",
                                    "value": action.pop(0),
                                    "start_time": state['timestep'] + 1})
            state = model.step(actions)
            
            agent.buffer.rewards.append(-state["energy"])
            agent.buffer.is_terminals.append(state["terminate"])
            total_energy += state["energy"]

        print(f"Episode: {ep}\t\tTotal energy: {total_energy}")
        log_f.write(f"Episode: {ep}\t\tTotal energy: {total_energy}\n")
        log_f.flush()
        agent.save(checkpoint_path)
        agent.update()

    log_f.close()
    print("Done")