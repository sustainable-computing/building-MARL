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
        default=10,
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
        default=1,
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
        default=200
    )
    parser.add_argument(
        '--prefix',
        help='Description',
        type=str,
        default=""
    )
    parser.add_argument(
        '--blind',
        help='Close blind or not, default is 0, do not close blind',
        type=int,
        default=0
    )
    parser.add_argument(
        '--ignore_zero_reward',
        help='Do not put the history of zero reward into the buffer.',
        type=int,
        default=1
    )
    
    args = parser.parse_args()
    np.random.RandomState(args.seed * 10)
    torch.manual_seed(args.seed * 10)
    
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
    os.makedirs(f"PPO_weights", exist_ok=True)
    optimal_path = None
    if not args.multi_agent:
        os.makedirs(f"PPO_weights/{args.prefix}SA_seed{args.seed}", exist_ok=True)
        checkpoint_path = f"PPO_weights/{args.prefix}SA_seed{args.seed}/PPO_seed{args.seed}.pth"
    elif args.train_on == -1:
        os.makedirs(f"PPO_weights/{args.prefix}MA_parallel_training_seed{args.seed}", exist_ok=True)
        checkpoint_path = f"PPO_weights/{args.prefix}MA_parallel_training_seed{args.seed}/agent"
        if args.diverse_training != 0:
            os.makedirs(f"PPO_weights/{args.prefix}MA_parallel_training_seed{args.seed}/{args.diverse_weight}", exist_ok=True)
            checkpoint_path = f"PPO_weights/{args.prefix}MA_parallel_training_seed{args.seed}/{args.diverse_weight}/agent"
            optimal_path = f"PPO_weights/{args.prefix}MA_parallel_training_seed{args.seed}/agent"
    else:
        os.makedirs(f"PPO_weights/{args.prefix}MA_rule_other_seed{args.seed}", exist_ok=True)
        checkpoint_path = f"PPO_weights/{args.prefix}MA_rule_other_seed{args.seed}/agent"
        if args.diverse_training != 0:
            os.makedirs(f"PPO_weights/{args.prefix}MA_rule_other_seed{args.seed}/{args.diverse_weight}", exist_ok=True)
            checkpoint_path = f"PPO_weights/{args.prefix}MA_rule_other_seed{args.seed}/{args.diverse_weight}/agent"
    
    log_path = f"{checkpoint_path[:checkpoint_path.rfind('/')]}/log_{args.seed}"
    log_f = open(f"{log_path}_{len(glob.glob(log_path + '_*'))}", "w+")

    # available_zones = ["SPACE1-1", "SPACE2-1", "SPACE3-1", "SPACE4-1", "SPACE5-1"]
    available_zones = ["Core_ZN", "Perimeter_ZN_1", "Perimeter_ZN_2", "Perimeter_ZN_3", "Perimeter_ZN_4"]
    airloops = {'Core_ZN': "PSZ-AC:1",
                'Perimeter_ZN_1': "PSZ-AC:2", 'Perimeter_ZN_2': "PSZ-AC:3",
                'Perimeter_ZN_3': "PSZ-AC:4", 'Perimeter_ZN_4': "PSZ-AC:5"}

    # Add state variables that we care about
    eplus_extra_states = {("Zone Air Relative Humidity", zone): f"{zone} humidity" for zone in available_zones}
    eplus_extra_states.update({("Air System Electric Energy", airloops[zone]): f"{zone} vav energy" for zone in available_zones})
    eplus_extra_states[('Site Outdoor Air Drybulb Temperature', 'Environment')] = "outdoor temperature"
    eplus_extra_states[('Site Direct Solar Radiation Rate per Area', 'Environment')] = "site solar radiation"
    eplus_extra_states[('Facility Total HVAC Electric Demand Power', 'Whole Building')] = "total hvac"

    model = Model(idf_file_name="./eplus_files/5ZoneAirCooled_electric.idf",
                  weather_file="./eplus_files/USA_CO_Denver-Aurora-Buckley.AFB_.724695_TMY3.epw",
                  eplus_naming_dict=eplus_extra_states,
                  tmp_idf_path=checkpoint_path[:checkpoint_path.rfind('/')])
    
    # Add them to the IDF file so we can retrieve them
    for key, _ in eplus_extra_states.items():
        model.add_configuration("Output:Variable",
                                {"Key Value": key[1], "Variable Name": key[0], "Reporting Frequency": "Timestep"})
    
    # Setup controls to all VAV boxes
    control_zones = available_zones[:]
    if args.multi_agent and args.train_on != -1:
        control_zones = [available_zones[args.train_on]]
    for zone in control_zones:
        model.add_configuration("Schedule:Constant",
                                {"Name": f"{zone} VAV Customized Schedule",
                                 "Schedule Type Limits Name": "Fraction",
                                 "Hourly Value": 0})
        model.edit_configuration(idf_header_name="AirTerminal:SingleDuct:VAV:NoReheat",
                                 identifier={"Name": f"{zone} Direct Air"},
                                 update_values={"Zone Minimum Air Flow Input Method": "Scheduled",
                                                "Minimum Air Flow Fraction Schedule Name": f"{zone} VAV Customized Schedule"})

    # Environment setup
    model.set_runperiod(*(30, 1991, args.start_month, 1))
    model.set_timestep(4)
    
    all_window = set()
    for windows in model.get_windows().values():
        all_window.update(windows)

    # Add blind
    if args.blind:
        model.set_blinds(sorted(list(all_window)), shading_control_type="AlwaysOn", setpoint=1)
    
    # Agent setup
    if not args.multi_agent:
        agent = PPO(5 + 5 + 1 + 1 + 5 + 1, # State dimension, temp + humidity + outdoor temp + solar + occupancy + hour
                    5, # Action dimension, 1 for each zone
                    args.lr_actor, args.lr_critic, args.gamma, args.k_epochs, args.eps_clip, has_continuous_action_space=True, action_std_init=0.6, 
                    device=device,
                    diverse_policies=list(), diverse_weight=0)
    else:
        num_rl_agent = 5 if args.train_on == -1 else 1
        if optimal_path is None:
            diverse_policies_temp = list()
        else:
            diverse_policies_temp = [f"{optimal_path}.pth"]
        agent = [PPO(1 + 1 + 1 + 1 + 1 + 1, # State dimension, own temperature + humidity + outdoor temp + solar + occupancy + hour
                     1, # Action dimension, 1 for each zone
                     0.003, 0.0005, 1, 10, 0.2, has_continuous_action_space=True, action_std_init=0.6, 
                     device=device,
                     diverse_policies=[f"{optimal_path}_{i}.pth"], diverse_weight=args.diverse_weight, diverse_increase=True) for i in range(num_rl_agent)]

    # Add to log visited (trained) states
    if not args.multi_agent:
        log_state = open(f"{log_path}_state", "w+")
    else:
        log_state = [open(f"{log_path}_state_agent_{i}", "w+") for i in range(5)]

    best_reward = -1

    for ep in range(args.episodes):
        state = model.reset()
        total_energy = state["total hvac"]
        while not model.is_terminate():
            # Transfer the state into the format of only selected states
            for zone in state["occupancy"]:
                state["occupancy"][zone] = 1 if state["occupancy"][zone] > 0 else 0
            agent_state = [state["outdoor temperature"], state["site solar radiation"], state["time"].hour]
            action = list()
            
            # log_f.write(f"{state}\n")
            single_agent_reward = 0
            for i, zone in enumerate(control_zones):
                if not args.multi_agent:
                    agent_state.append(state[f"{zone} humidity"])
                    agent_state.append(state["temperature"][zone])
                    agent_state.append(state["occupancy"][zone])
                    single_agent_reward += -state[f"{zone} vav energy"]
                else:
                    agent_state_i = agent_state + [state[f"{zone} humidity"], state["temperature"][zone], state["occupancy"][zone]]
                    action.append(agent[i].select_action(agent_state_i))
                    log_state[i].write(f"{agent_state_i}")

            # Get action and round to 0.1~1
            if not args.multi_agent:
                action = agent.select_action(agent_state)
                log_state.write(f"{agent_state},{single_agent_reward}\n")
            else:
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
            
            if not args.multi_agent:
                agent.buffer.rewards.append(single_agent_reward)
                agent.buffer.is_terminals.append(state["terminate"])
            else:
                for i, zone in enumerate(control_zones):
                    agent[i].buffer.rewards.append(-state[f"{zone} vav energy"])  # -state[f"{airloops[zone]} energy"]
                    agent[i].buffer.is_terminals.append(state["terminate"])
                    log_state[i].write(f",{-state[f'{zone} vav energy']}\n")
                    
                    if args.ignore_zero_reward and -state[f"{zone} vav energy"] == 0:
                        agent[i].buffer.remove_last()

            total_energy += state["total hvac"]

        print(f"[{datetime.datetime.now()}]Episode: {ep}\t\tTotal energy: {total_energy}")
        log_f.write(f"[{datetime.datetime.now()}]Episode: {ep}\t\tTotal energy: {total_energy}\n")
        log_f.flush()

        # Policy save and learn
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
                if best_reward == -1 or int(best_reward) >= int(total_energy):
                    agent[i].save(f"{checkpoint_path}_{i}.pth")
                    best_reward = total_energy

    log_f.close()
    if not args.multi_agent:
        log_state.close()
    else:
        for i in range(len(agent)):
            log_state[i].close()
    print("Done")
