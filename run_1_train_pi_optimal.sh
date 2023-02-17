#!/usr/bin/bash

MONTH=(1)
IGNORES=(0 1)
BLINDS=(0 1)

for start_month in ${MONTH[@]} ; do
  for ignore in ${IGNORES[@]} ; do
    for blind in ${BLINDS[@]} ; do
      export ignore blind
      sbatch run_1_train_pi_optimal_sbatch.sh
    done
  done
done