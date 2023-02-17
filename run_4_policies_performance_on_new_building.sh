#!/usr/bin/bash

STARTS=(0 50 100 150 200 250 300 350 400 450 500 550 600 650 700 750 800 850)


for start in ${STARTS[@]} ; do
  export start
  sbatch run_4_policies_performance_on_new_building_b_denver.sh
  sbatch run_4_policies_performance_on_new_building_b_sf.sh
  sbatch run_4_policies_performance_on_new_building_c.sh
done