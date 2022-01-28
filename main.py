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
    parser.add_argument(
        '--multi_agent',
        help='Set to 1 if enable multi agent, else 0',
        type=int,
        default=0,
    )
    parser.add_argument(
        '--train_on',
        help='Set to -1 for parallel training, 0-14 otherwise indicating which zone to train, other zones will use rule-based controller',
        type=int,
        default=-1
    )
    parser.add_argument(
        '--diverse_training',
        help='Set to an integer for diverse training, where all other agents are previous policies. 0 to turn off',
        type=int,
        default=0
    )
    parser.add_argument(
        '--diverse_weight',
        help='Set the weight for diversity loss',
        type=float,
        default=0.01
    )
    parser.add_argument(
        '--start_month',
        help='Set the starting month',
        type=int,
        default=7
    )
    parser.add_argument(
        '--std_decay_period',
        help='Set the number of episode to decay std',
        type=int,
        default=100
    )
    parser.add_argument(
        '--prefix',
        help='Description',
        type=str,
        default=""
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
    os.makedirs(f"PPO_weights", exist_ok=True)
    if not args.multi_agent:
        os.makedirs(f"PPO_weights/{args.prefix}single_agent_{args.seed}", exist_ok=True)
        checkpoint_path = f"PPO_weights/{args.prefix}single_agent_{args.seed}/PPO_{args.seed}_{run_num}.pth"
    elif args.train_on == -1:
        os.makedirs(f"PPO_weights/{args.prefix}PPO_{args.seed}_{run_num}", exist_ok=True)
        checkpoint_path = f"PPO_weights/{args.prefix}PPO_{args.seed}_{run_num}/agent"
    else:
        os.makedirs(f"PPO_weights/{args.prefix}PPO_{args.seed}_{run_num}_single_env_training", exist_ok=True)
        checkpoint_path = f"PPO_weights/{args.prefix}PPO_{args.seed}_{run_num}_single_env_training/agent"
        if args.diverse_training != 0:
            os.makedirs(f"PPO_weights/{args.prefix}PPO_{args.seed}_{run_num}_single_env_training/{args.diverse_weight}", exist_ok=True)
            checkpoint_path = f"PPO_weights/{args.prefix}PPO_{args.seed}_{run_num}_single_env_training/{args.diverse_weight}/agent"
    
    log_path = f"{checkpoint_path[:checkpoint_path.rfind('/')]}/log_{args.seed}_{run_num}"
    log_f = open(f"{log_path}_{len(glob.glob(log_path + '_*'))}", "w+")

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
                  tmp_idf_path=checkpoint_path[:checkpoint_path.rfind('/')])
    
    # Add them to the IDF file so we can retrieve them
    for key, _ in eplus_extra_states.items():
        model.add_configuration("Output:Variable",
                                {"Key Value": key[1], "Variable Name": key[0], "Reporting Frequency": "Timestep"})
    
    # Setup controls to all VAV boxes
    control_zones = available_zones[3:]
    if args.multi_agent and args.train_on != -1:
        control_zones = [available_zones[3 + args.train_on]]
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
    model.set_runperiod(*(30, 1991, args.start_month, 1))
    model.set_timestep(4)
    
    # Agent setup
    # TODO: add single-agent diversity?
    if not args.multi_agent:
        agent = PPO(15 + 15 + 1 + 1 + 15 + 1, # State dimension, temp + humidity + outdoor temp + solar + occupancy + hour
                    15, # Action dimension, 1 for each zone
                    args.lr_actor, args.lr_critic, args.gamma, args.k_epochs, args.eps_clip, has_continuous_action_space=True, action_std_init=0.6, 
                    device=device,
                    diverse_policies=list(), diverse_weight=0)
    else:
        num_rl_agent = 15 if args.train_on == -1 else 1
        existing_policies = list()
        if args.train_on == -1 and args.diverse_training != 0:
            existing_policies = list(glob.glob(f"PPO_weights/PPO_{args.seed}_{run_num}_single_env_training/agent_*.pth"))
        agent = [PPO(1 + 1 + 1 + 1 + 1 + 1, # State dimension, own temperature + humidity + outdoor temp + solar + occupancy + hour
                     1, # Action dimension, 1 for each zone
                     0.003, 0.0005, 1, 10, 0.2, has_continuous_action_space=True, action_std_init=0.6, 
                     device=device,
                     diverse_policies=existing_policies, diverse_weight=args.diverse_weight, diverse_increase=True) for _ in range(num_rl_agent)]
        
    best_reward = -1
    
    for ep in range(args.episodes):
        state = model.reset()
        total_energy = state["total hvac"]
        while not model.is_terminate():
            # Transfer the state into the format of only selected states
            agent_state = [state["outdoor temperature"], state["site solar radiation"], state["time"].hour]
            action = list()
            
            single_agent_reward = 0
            for i, zone in enumerate(control_zones):
                if not args.multi_agent:
                    agent_state.append(state[f"{zone} humidity"])
                    agent_state.append(state["temperature"][zone])
                    agent_state.append(state["occupancy"][zone])
                    single_agent_reward += -state[f"{zone} vav energy"]
                else:
                    action.append(agent[i].select_action(agent_state + [state[f"{zone} humidity"], state["temperature"][zone], state["occupancy"][zone]]))
                    agent[i].buffer.rewards.append(-state[f"{zone} vav energy"])  # -state[f"{airloops[zone]} energy"]
                    agent[i].buffer.is_terminals.append(state["terminate"])
            
            # Get action and round to 0~1
            if not args.multi_agent:
                action = agent.select_action(agent_state)
            else:
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
            
            if not args.multi_agent:
                agent.buffer.rewards.append(single_agent_reward)
                agent.buffer.is_terminals.append(state["terminate"])
            total_energy += state["total hvac"]

        print(f"[{datetime.datetime.now()}]Episode: {ep}\t\tTotal energy: {total_energy}")
        log_f.write(f"[{datetime.datetime.now()}]Episode: {ep}\t\tTotal energy: {total_energy}\n")
        log_f.flush()
        
        if not args.multi_agent:
            if ((ep + 1) % args.std_decay_period) == 0:
                agent.decay_action_std(0.02, 0.1)
            agent.update()
            if best_reward == -1 or best_reward > total_energy:
                agent.save(checkpoint_path)
                best_reward = total_energy
        else:
            for i in range(len(agent)):
                if ((ep + 1) % args.std_decay_period) == 0:
                    agent[i].decay_action_std(0.02, 0.1)
                agent[i].update()
                if best_reward == -1 or best_reward > total_energy:
                    if args.diverse_training != 0:
                        agent[i].save(f"{checkpoint_path}_{args.diverse_training}.pth")
                    else:
                        agent[i].save(f"{checkpoint_path}_{i}.pth")
                    best_reward = total_energy

    log_f.close()
    print("Done")