#!/usr/bin/bash

SAS=(0 1)
FLS=(0)
IZS=(0)
SEDPRX=(0 1 2)

for seed_prefix in ${SEDPRX[@]} ; do
  for sa in ${SAS[@]} ; do
    for fl in ${FLS[@]} ; do
      for iz in ${IZS[@]} ; do
        export sa fl iz seed_prefix
        sbatch run_6_train_from_scratch_on_new_building_b_denver.sh
        sbatch run_6_train_from_scratch_on_new_building_b_sf.sh
        sbatch run_6_train_from_scratch_on_new_building_c.sh
      done
    done
  done
done