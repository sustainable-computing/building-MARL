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
    # set device to cpu or cuda
    device = torch.device('cpu')
    print("Device set to: cpu")

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
    
    # Facility Total HVAC Electric Demand Power
    # Heating Coil Electric Power
    # eplus_meter_dict = {f"Heating:EnergyTransfer:Zone:{zone}": f"{zone} heating energy" for zone in available_zones}
    # eplus_meter_dict.update({f"Cooling:EnergyTransfer:Zone:{zone}": f"{zone} cooling energy" for zone in available_zones})
    # eplus_meter_dict.update({f"EnergyTransfer:Zone:{zone}": f"{zone} total transfer energy" for zone in available_zones})
    # eplus_meter_dict.update({f"Electricity:Zone:{zone}": f"{zone} total energy" for zone in available_zones})
    # eplus_meter_dict["Electricity:HVAC"] = "total hvac report"
    # eplus_meter_dict["EnergyTransfer:HVAC"] = "total EnergyTransfer report"
    
    # eplus_meter_dict["EnergyTransfer:Facility"] = "Facility report"
    # eplus_meter_dict["EnergyTransfer:Building"] = "Building report"

    model = Model(idf_file_name="./eplus_files/OfficeMedium_Denver.idf",
                  weather_file="./eplus_files/USA_CO_Denver-Aurora-Buckley.AFB_.724695_TMY3.epw",
                  eplus_naming_dict=eplus_extra_states,
                #   eplus_meter_dict=eplus_meter_dict,
                  )
    
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
    model.set_runperiod(*(1, 1991, 7, 1))
    model.set_timestep(1)
    
    # Agent setup
    if False:
        agent = PPO(15 + 15 + 1 + 1 + 15 + 1, # State dimension, temp + humidity + outdoor temp + solar + occupancy + hour
                    15, # Action dimension, 1 for each zone
                    0.003, 0.0005, 1, 10, 0.2, has_continuous_action_space=True, action_std_init=0.6, 
                    device=device,
                    diverse_policies=list(), diverse_weight=0)
    else:
        agent = [PPO(1 + 1 + 1 + 1 + 1 + 1, # State dimension, own temperature + humidity + outdoor temp + solar + occupancy + hour
                     1, # Action dimension, 1 for each zone
                     0.003, 0.0005, 1, 10, 0.2, has_continuous_action_space=True, action_std_init=0.6, 
                     device=device,
                     diverse_policies=list(), diverse_weight=0) for _ in range(15)]
    
    for ep in range(1):
        state = model.reset()
        total_energy = state["energy"]
        while not model.is_terminate():
            # Transfer the state into the format of only selected states
            agent_state = [state["outdoor temperature"], state["site solar radiation"], state["time"].hour]

            action = list()
            for i, zone in enumerate(available_zones[3:]):
                if True:
                    action.append(agent[i].select_action(agent_state + [state[f"{zone} humidity"], state["temperature"][zone], state["occupancy"][zone]]))
                    agent[i].buffer.rewards.append(-state[f"{zone} vav energy"] + -state[f"{airloops[zone]} energy"])
                    agent[i].buffer.is_terminals.append(state["terminate"])
                else:
                    agent_state.append(state[f"{zone} humidity"])
                    agent_state.append(state["temperature"][zone])
                    agent_state.append(state["occupancy"][zone])
            
            # Get action and round to 0~1
            if False:
                action = agent.select_action(agent_state)
            else:
                action = np.array(action)
            action = list(1/(1 + np.exp(-action)))
            
            actions = list()
            for i, zone in enumerate(available_zones[3:]):
                actions.append({"priority": 0,
                                "component_type": "Schedule:Constant",
                                "control_type": "Schedule Value",
                                "actuator_key": f"{zone} VAV Customized Schedule",
                                "value": action[i],
                                "start_time": state['timestep'] + 1})
            state = model.step(actions)
            
            if False:
                agent.buffer.rewards.append(-state["total hvac"])
                agent.buffer.is_terminals.append(state["terminate"])
            total_energy += state["total hvac"]

        print(f"Episode: {ep}\t\tTotal energy: {total_energy}")
        # for key in state:
        #     print(key, state[key])
        if False:
            agent.update()
        else:
            for sub_agent in agent:
                sub_agent.update()

    print("Done")