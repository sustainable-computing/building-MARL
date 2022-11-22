from local_setting import *
import sys
import os
import random
from ppo import PPO
import argparse
import torch
import numpy as np
import glob
import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.insert(0, file_path)
from cobs import Model

Model.set_energyplus_folder(energyplus_location)
import pandas as pd

if __name__ == '__main__':
    # analyze()
    # Setup run parameters
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--evaluate_on',
        help='0-14 indicating which zone to train, other zones will use rule-based controller',
        type=int,
        default=14
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
    parser.add_argument(
        '--change_rotation_location',
        help='1 for california and 45 deg rotation',
        type=int,
        default=0
    )
    parser.add_argument(
        '--doee',
        help='1 for doee',
        type=int,
        default=0
    )

    args = parser.parse_args()

    print("============================================================================================")
    # set device to cpu or cuda
    device = torch.device('cpu')

    weight_mapping = {"1e0": "1.0", "1e-1": "0.1", "1e1": "10.0",
                      "1e0_2": "1.0_2", "1e-1_2": "0.1_2", "1e1_2": "10.0_2"}

    if args.doee:
        initial_state = {'timestep': 0, 'time': datetime.datetime(1999, 1, 1, 0, 15),
                         'temperature': {'Amphitheater': 18.337235735030927, 'Conferance': 18.369132674627814,
                                         'Corridor1': 22.001236343367594, 'Corridor2': 21.999261031553086,
                                         'Corridor3': 21.999144472749858, 'CorridorB': 22.00081670577097,
                                         'CorridorG': 21.999413942251223, 'Engine Room': 13.794749564227782,
                                         'Lab': 19.042237536663563, 'Library': 17.911334685788027,
                                         'North-1': 19.172404032545813, 'North-2': 19.08172618971252,
                                         'North-3': 19.291900665819075, 'North-G': 18.681938131436787,
                                         'Plenum1': 19.195598583894107, 'Plenum2': 18.825091491275025,
                                         'Plenum3': 14.95218847968496, 'PlenumAmph': 17.199897419228922,
                                         'PlenumB': 18.28148298226102, 'PlenumG': 18.93663704221402,
                                         'South-1': 19.40641525097876, 'South-2': 19.499355229479512,
                                         'South-3': 22.011122935672933, 'South-GF': 18.91008681700714,
                                         'Stairs': 13.695673026021913, 'WC-AHU-1N': 19.276986579858775,
                                         'WC-AHU-1S': 18.971120000813425, 'WC-AHU-2N': 19.711579953811516,
                                         'WC-AHU-2S': 19.00227382746589, 'WC-AHU-3N': 18.065565332028317,
                                         'WC-AHU-3S': 17.46859790572499, 'WC-AHU-BN': 17.355666333585194,
                                         'WC-AHU-BS': 17.767716985379508, 'WC-AHU-GN': 18.31313965731881,
                                         'WC-AHU-GS': 18.45249747321873},
                         'occupancy': {'Amphitheater': 0, 'Conferance': 0.0, 'Corridor1': 0.0, 'Corridor2': 0.0,
                                       'Corridor3': 0.0, 'CorridorB': 0.0, 'CorridorG': 0.0, 'Engine Room': 0.0,
                                       'Lab': 0, 'Library': 0, 'North-1': 0, 'North-2': 0, 'North-3': 0, 'North-G': 0,
                                       'South-1': 0, 'South-2': 0, 'South-3': 0, 'South-GF': 0, 'Stairs': 0.0},
                         'terminate': False, 'energy': 0.0, 'Amphitheater humidity': 37.956364745689974,
                         'Lab humidity': 36.31268198138177, 'Library humidity': 38.9704793940533,
                         'North-1 humidity': 36.007836729652674, 'North-2 humidity': 36.212616567059705,
                         'North-3 humidity': 35.716467735324976, 'North-G humidity': 37.13392670638036,
                         'South-1 humidity': 35.48153588571067, 'South-2 humidity': 35.276023215651364,
                         'South-3 humidity': 30.135841494311634, 'South-GF humidity': 36.60442699533386,
                         'Amphitheater vav heating energy': 8203.923241638284,
                         'Lab vav heating energy': 9157.114213068686, 'Library vav heating energy': 10013.64189277092,
                         'North-1 vav heating energy': 10389.339654124755,
                         'North-2 vav heating energy': 10284.054722870542,
                         'North-3 vav heating energy': 12890.675926294283,
                         'North-G vav heating energy': 9935.363369595781,
                         'South-1 vav heating energy': 10722.947083285382,
                         'South-2 vav heating energy': 10952.572674475434,
                         'South-3 vav heating energy': 18906.915105086067,
                         'South-GF vav heating energy': 10239.138419322144, 'Amphitheater vav cooling energy': 0.0,
                         'Lab vav cooling energy': 0.0, 'Library vav cooling energy': 0.0,
                         'North-1 vav cooling energy': 0.0, 'North-2 vav cooling energy': 0.0,
                         'North-3 vav cooling energy': 0.0, 'North-G vav cooling energy': 0.0,
                         'South-1 vav cooling energy': 0.0, 'South-2 vav cooling energy': 0.0,
                         'South-3 vav cooling energy': 0.0, 'South-GF vav cooling energy': 0.0,
                         'Amphitheater position': 0.3, 'Lab position': 0.3, 'Library position': 0.3,
                         'North-1 position': 0.3, 'North-2 position': 0.3, 'North-3 position': 0.3,
                         'North-G position': 0.3, 'South-1 position': 0.3, 'South-2 position': 0.3,
                         'South-3 position': 0.517814880621679, 'South-GF position': 0.3, 'outdoor temperature': 6.825,
                         'site solar radiation': 0.0, 'total hvac': 6987.887899656726,
                         'Amphitheater vav energy': 8203.923241638284, 'Lab vav energy': 9157.114213068686,
                         'Library vav energy': 10013.64189277092, 'North-1 vav energy': 10389.339654124755,
                         'North-2 vav energy': 10284.054722870542, 'North-3 vav energy': 12890.675926294283,
                         'North-G vav energy': 9935.363369595781, 'South-1 vav energy': 10722.947083285382,
                         'South-2 vav energy': 10952.572674475434, 'South-3 vav energy': 18906.915105086067,
                         'South-GF vav energy': 10239.138419322144}
    elif args.change_rotation_location:
        initial_state = {'timestep': 0, 'time': datetime.datetime(1999, 1, 1, 0, 15),
                         'temperature': {'Core_bottom': 22.38439002984805, 'TopFloor_Plenum': 19.24731225580394,
                                         'MidFloor_Plenum': 21.655832762034745, 'FirstFloor_Plenum': 21.74907554606701,
                                         'Core_mid': 22.56821725166586, 'Core_top': 21.56012416265049,
                                         'Perimeter_top_ZN_3': 19.556145135739616,
                                         'Perimeter_top_ZN_2': 20.473151073853693,
                                         'Perimeter_top_ZN_1': 20.669246200348944,
                                         'Perimeter_top_ZN_4': 19.49883462005278,
                                         'Perimeter_bot_ZN_3': 19.873416501507567,
                                         'Perimeter_bot_ZN_2': 20.595710444774586,
                                         'Perimeter_bot_ZN_1': 20.804703526301694,
                                         'Perimeter_bot_ZN_4': 19.83515249669134,
                                         'Perimeter_mid_ZN_3': 20.344607511848746,
                                         'Perimeter_mid_ZN_2': 21.234468760806752,
                                         'Perimeter_mid_ZN_1': 21.454262590138487,
                                         'Perimeter_mid_ZN_4': 20.286067534776173},
                         'occupancy': {'Core_bottom': 0, 'Core_mid': 0, 'Core_top': 0, 'Perimeter_top_ZN_3': 0,
                                       'Perimeter_top_ZN_2': 0, 'Perimeter_top_ZN_1': 0, 'Perimeter_top_ZN_4': 0,
                                       'Perimeter_bot_ZN_3': 0, 'Perimeter_bot_ZN_2': 0, 'Perimeter_bot_ZN_1': 0,
                                       'Perimeter_bot_ZN_4': 0, 'Perimeter_mid_ZN_3': 0, 'Perimeter_mid_ZN_2': 0,
                                       'Perimeter_mid_ZN_1': 0, 'Perimeter_mid_ZN_4': 0}, 'terminate': False,
                         'energy': 0.0, 'TopFloor_Plenum humidity': 41.905158399774336,
                         'MidFloor_Plenum humidity': 36.72441371402642, 'FirstFloor_Plenum humidity': 36.71097611273987,
                         'Core_top humidity': 39.80342116425452, 'Core_mid humidity': 36.967023894843834,
                         'Core_bottom humidity': 36.644641224641624, 'Perimeter_top_ZN_3 humidity': 38.16837104961714,
                         'Perimeter_top_ZN_2 humidity': 36.3378023766978,
                         'Perimeter_top_ZN_1 humidity': 35.671428771715654,
                         'Perimeter_top_ZN_4 humidity': 38.384133324964424,
                         'Perimeter_bot_ZN_3 humidity': 39.115208701929696,
                         'Perimeter_bot_ZN_2 humidity': 38.066408670936994,
                         'Perimeter_bot_ZN_1 humidity': 37.12448864545414,
                         'Perimeter_bot_ZN_4 humidity': 39.60047361716256,
                         'Perimeter_mid_ZN_3 humidity': 37.06923231968373,
                         'Perimeter_mid_ZN_2 humidity': 35.34033337318468,
                         'Perimeter_mid_ZN_1 humidity': 34.72519136679645,
                         'Perimeter_mid_ZN_4 humidity': 37.33324245152082, 'Core_top vav energy': 0.0,
                         'Core_mid vav energy': 0.0, 'Core_bottom vav energy': 0.0,
                         'Perimeter_top_ZN_3 vav energy': 0.0, 'Perimeter_top_ZN_2 vav energy': 0.0,
                         'Perimeter_top_ZN_1 vav energy': 0.0, 'Perimeter_top_ZN_4 vav energy': 0.0,
                         'Perimeter_bot_ZN_3 vav energy': 0.0, 'Perimeter_bot_ZN_2 vav energy': 0.0,
                         'Perimeter_bot_ZN_1 vav energy': 0.0, 'Perimeter_bot_ZN_4 vav energy': 0.0,
                         'Perimeter_mid_ZN_3 vav energy': 0.0, 'Perimeter_mid_ZN_2 vav energy': 0.0,
                         'Perimeter_mid_ZN_1 vav energy': 0.0, 'Perimeter_mid_ZN_4 vav energy': 0.0,
                         'Core_top position': 0.0, 'Core_mid position': 0.0, 'Core_bottom position': 0.0,
                         'Perimeter_top_ZN_3 position': 0.0, 'Perimeter_top_ZN_2 position': 0.0,
                         'Perimeter_top_ZN_1 position': 0.0, 'Perimeter_top_ZN_4 position': 0.0,
                         'Perimeter_bot_ZN_3 position': 0.0, 'Perimeter_bot_ZN_2 position': 0.0,
                         'Perimeter_bot_ZN_1 position': 0.0, 'Perimeter_bot_ZN_4 position': 0.0,
                         'Perimeter_mid_ZN_3 position': 0.0, 'Perimeter_mid_ZN_2 position': 0.0,
                         'Perimeter_mid_ZN_1 position': 0.0, 'Perimeter_mid_ZN_4 position': 0.0,
                         'PACU_VAV_bot energy': 0.0, 'PACU_VAV_top energy': 0.0, 'PACU_VAV_mid energy': 0.0,
                         'outdoor temperature': 6.825, 'site solar radiation': 0.0, 'total hvac': 168.34104864229795,
                         'operations availability': 0.0}
    else:
        initial_state = {'timestep': 0, 'time': datetime.datetime(2005, 1, 1, 0, 15),
                         'temperature': {'Core_bottom': 21.581801996249453, 'TopFloor_Plenum': 15.924248458088547,
                                         'MidFloor_Plenum': 19.480189779312457, 'FirstFloor_Plenum': 20.42203554607113,
                                         'Core_mid': 20.697094368574586, 'Core_top': 18.776679156600338,
                                         'Perimeter_top_ZN_3': 16.395610201114657,
                                         'Perimeter_top_ZN_2': 16.756877596648764,
                                         'Perimeter_top_ZN_1': 18.41755710231676,
                                         'Perimeter_top_ZN_4': 17.01708112174805,
                                         'Perimeter_bot_ZN_3': 18.487414356937187,
                                         'Perimeter_bot_ZN_2': 18.687784483193713,
                                         'Perimeter_bot_ZN_1': 19.994621927953965,
                                         'Perimeter_bot_ZN_4': 18.926801503829996,
                                         'Perimeter_mid_ZN_3': 17.9945033919463,
                                         'Perimeter_mid_ZN_2': 18.289552814209888,
                                         'Perimeter_mid_ZN_1': 19.95251413676269,
                                         'Perimeter_mid_ZN_4': 18.550744155006033},
                         'occupancy': {'Core_bottom': 0.0, 'Core_mid': 0.0, 'Core_top': 0.0, 'Perimeter_top_ZN_3': 0.0,
                                       'Perimeter_top_ZN_2': 0.0, 'Perimeter_top_ZN_1': 0.0, 'Perimeter_top_ZN_4': 0.0,
                                       'Perimeter_bot_ZN_3': 0.0, 'Perimeter_bot_ZN_2': 0.0, 'Perimeter_bot_ZN_1': 0.0,
                                       'Perimeter_bot_ZN_4': 0.0, 'Perimeter_mid_ZN_3': 0.0, 'Perimeter_mid_ZN_2': 0.0,
                                       'Perimeter_mid_ZN_1': 0.0, 'Perimeter_mid_ZN_4': 0.0},
                         'terminate': False, 'energy': 0.0, 'TopFloor_Plenum humidity': 24.00868936387423,
                         'MidFloor_Plenum humidity': 22.169133960726555,
                         'FirstFloor_Plenum humidity': 20.523208512457757,
                         'Core_top humidity': 45.43116507702326, 'Core_mid humidity': 23.90098141680685,
                         'Core_bottom humidity': 20.316350771686384, 'Perimeter_top_ZN_3 humidity': 25.197673135998162,
                         'Perimeter_top_ZN_2 humidity': 25.295862021247505,
                         'Perimeter_top_ZN_1 humidity': 22.823620936950736,
                         'Perimeter_top_ZN_4 humidity': 24.914829374069654,
                         'Perimeter_bot_ZN_3 humidity': 23.245618681203105,
                         'Perimeter_bot_ZN_2 humidity': 22.92022237776397,
                         'Perimeter_bot_ZN_1 humidity': 21.373485323344294,
                         'Perimeter_bot_ZN_4 humidity': 22.677383048168238,
                         'Perimeter_mid_ZN_3 humidity': 23.33201209845985,
                         'Perimeter_mid_ZN_2 humidity': 23.359061724994408,
                         'Perimeter_mid_ZN_1 humidity': 21.172944348273223,
                         'Perimeter_mid_ZN_4 humidity': 23.072737770819, 'Core_top vav energy': 0.0,
                         'Core_mid vav energy': 0.0,
                         'Core_bottom vav energy': 0.0, 'Perimeter_top_ZN_3 vav energy': 0.0,
                         'Perimeter_top_ZN_2 vav energy': 0.0,
                         'Perimeter_top_ZN_1 vav energy': 0.0, 'Perimeter_top_ZN_4 vav energy': 0.0,
                         'Perimeter_bot_ZN_3 vav energy': 0.0,
                         'Perimeter_bot_ZN_2 vav energy': 0.0, 'Perimeter_bot_ZN_1 vav energy': 0.0,
                         'Perimeter_bot_ZN_4 vav energy': 0.0,
                         'Perimeter_mid_ZN_3 vav energy': 0.0, 'Perimeter_mid_ZN_2 vav energy': 0.0,
                         'Perimeter_mid_ZN_1 vav energy': 0.0,
                         'Perimeter_mid_ZN_4 vav energy': 0.0, 'PACU_VAV_bot energy': 0.0, 'PACU_VAV_mid energy': 0.0,
                         'PACU_VAV_top energy': 0.0,
                         'outdoor temperature': 0.75, 'site solar radiation': 0.0, 'total hvac': 168.34104864229795}

    if args.doee:
        available_zones = ["Amphitheater", "Lab", "Library",
                           "North-1", "North-2", "North-3", "North-G",
                           "South-1", "South-2", "South-3", "South-GF"]
        test_zone = available_zones[args.evaluate_on]
    else:
        available_zones = ['TopFloor_Plenum', 'MidFloor_Plenum', 'FirstFloor_Plenum',
                           'Core_top', 'Core_mid', 'Core_bottom',
                           'Perimeter_top_ZN_3', 'Perimeter_top_ZN_2', 'Perimeter_top_ZN_1', 'Perimeter_top_ZN_4',
                           'Perimeter_bot_ZN_3', 'Perimeter_bot_ZN_2', 'Perimeter_bot_ZN_1', 'Perimeter_bot_ZN_4',
                           'Perimeter_mid_ZN_3', 'Perimeter_mid_ZN_2', 'Perimeter_mid_ZN_1', 'Perimeter_mid_ZN_4']
        test_zone = available_zones[3 + args.evaluate_on]

    agent = PPO(1 + 1 + 1 + 1 + 1 + 1,
                # State dimension, own temperature + humidity + outdoor temp + solar + occupancy + hour
                1,  # Action dimension, 1 for each zone
                0.003, 0.0005, 1, 10, 0.2, has_continuous_action_space=True, action_std_init=0.2,
                device=device,
                diverse_policies=list(), diverse_weight=0, diverse_increase=True)
    agent_save_paths = sorted(list(glob.glob("policy_library/**.pth")))

    error_count = 0

    x = list()
    y = list()
    num_random_points = 10
    np.random.seed(1080)

    state_limit = {
        0: (0, 30),  # outdoor temperature
        1: (0, 30),  # beam solar
        2: (0, 12),  # time
        3: (0, 1),  # humidity
        4: (15, 30),  # indoor temperature
        5: (0, 1),  # occupancy
    }

    data = [["name", "optimal", "init_action"] + [f"train_state_{i}" for i in range(6)] + [f"p{i}" for i in
                                                                                           range(num_random_points)]]
    temp_states = list()

    for agent_path in tqdm(agent_save_paths):
        agent.load(agent_path)
        try:
            agent.select_action([10, 1, 1, 1, 1, 1])
        except ValueError:
            continue

        row = [agent_path, "-1"]

        occupancy = 1 if initial_state["occupancy"][test_zone] > 0 else 0
        # Transfer the state into the format of only selected states
        init_state = [initial_state["outdoor temperature"],
                      initial_state["site solar radiation"],
                      initial_state["time"].hour,
                      initial_state[f"{test_zone} humidity"],
                      initial_state["temperature"][test_zone],
                      occupancy]

        action = agent.get_mean(init_state)
        action = np.array(action)
        action = 0.9 / (1 + np.exp(-action)) + 0.1

        row.append(action[0])

        path_name = os.path.basename(agent_path)
        ignore = path_name[1]
        blind = 1 if "blind" in path_name else 0
        if 'e' not in path_name:
            weight = ''
        else:
            weight = weight_mapping[path_name[path_name.index("1e"):-4].strip("_blind")] + '/'
        agent_training_zone = path_name[path_name.index('_') + 1]
        state_history = f"PPO_weights/5_zone_ignore_{ignore}_blind_{blind}_OPTIMAL_MA_parallel_training_seed{path_name[:3]}/{weight}log_{path_name[:3]}_state_agent_{agent_training_zone}"
        values = [[], [], [], [], [], [], []]
        counter = 0
        with open(state_history, 'r') as history:
            for line in history:
                counter += 1
        selected_states = np.random.choice(range(counter), num_random_points)
        with open(state_history, 'r') as history:
            for line in history:
                counter -= 1
                line = line.split(',')
                line[0] = line[0][1:]
                line[5] = line[5][:-1]
                if len(temp_states) != num_random_points and counter in selected_states:
                    temp_states.append([float(v) for v in line[:-1]])

                for i in range(3, len(line)):
                    values[i].append(float(line[i]))

        values.pop(0)
        values.pop(0)
        values.pop(0)
        values.pop(-2)
        for v in values:
            row.append(np.mean(v))
            row.append(np.std(v))

        for i in range(num_random_points):
            temp_state = temp_states[i]
            a = 0.9 / (1 + np.exp(-np.array(agent.get_mean(temp_state)))) + 0.1
            row.append(a[0])

        agent.buffer.clear()
        data.append(row)

    os.makedirs(f"policy_library/transfer/zeroshot_newloc_{args.change_rotation_location}_doee_{args.doee}",
                exist_ok=True)
    with open(
            f"policy_library/transfer/zeroshot_newloc_{args.change_rotation_location}_doee_{args.doee}/zero-shot_zone_{args.evaluate_on}.csv",
            "w") as outfile:
        for row in data:
            outfile.write(",".join(list(map(str, row))) + "\n")

    print("Done")
