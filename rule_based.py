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
    log_f = open(f"log_rule_based", "w+")
    
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
                  eplus_naming_dict=eplus_extra_states)
    
    # Add them to the IDF file so we can retrieve them
    for key, _ in eplus_extra_states.items():
        model.add_configuration("Output:Variable",
                                {"Key Value": key[1], "Variable Name": key[0], "Reporting Frequency": "Timestep"})

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
    model.set_runperiod(*(30, 1991, 7, 1))
    model.set_timestep(4)
    
    for ep in range(5):
        state = model.reset()
        total_energy = state["total hvac"]
        while not model.is_terminate():
            actions = list()
            for i, zone in enumerate(control_zones):
                actions.append({"priority": 0,
                                "component_type": "Schedule:Constant",
                                "control_type": "Schedule Value",
                                "actuator_key": f"{zone} VAV Customized Schedule",
                                "value": 0.3,
                                "start_time": state['timestep'] + 1})
            state = model.step(actions)
            total_energy += state["total hvac"]

        print(f"[{datetime.datetime.now()}]Episode: {ep}\t\tTotal energy: {total_energy}")
        log_f.write(f"[{datetime.datetime.now()}]Episode: {ep}\t\tTotal energy: {total_energy}\n")
        log_f.flush()

    log_f.close()
    print("Done")