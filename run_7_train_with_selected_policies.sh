#!/usr/bin/bash

# ============================ clusters =======================
SCRATCH_OPT=(0)
CLUSTERS_OPT=(6)
SEDPRX=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14)


for scratch in ${SCRATCH_OPT[@]} ; do
  for clusters in ${CLUSTERS_OPT[@]} ; do
    for seed_prefix in ${SEDPRX[@]} ; do
      export scratch clusters seed_prefix
      sbatch run_7_train_with_selected_policies_b_denver.sh
      sbatch run_7_train_with_selected_policies_b_sf.sh
      sbatch run_7_train_with_selected_policies_c.sh
    done
  done
done