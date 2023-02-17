#!/usr/bin/bash

MONTH=(1)
IGNORES=(0 1)
WEIGHTS=(10 1 0.1)
BLINDS=(0 1)

for start_month in ${MONTH[@]} ; do
  for ignore in ${IGNORES[@]} ; do
    for weight in ${WEIGHTS[@]} ; do
      for blind in ${BLINDS[@]} ; do
        export start_month ignore weight blind
        sbatch run_1_train_with_policy_diversity_sbatch.sh
      done
    done
  done
done