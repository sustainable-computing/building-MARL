#!/bin/bash

SEEDS=(1 2 3 4)
TRAIN_ON=(1 11 12 13 14)
gpu=0
for seed in ${SEEDS[@]} ; do
    for train_zone in ${TRAIN_ON[@]} ; do
        # tmux kill-session -t "bash-session-$train_zone$seed"
        tmux new-session -d -s "bash-session-$train_zone$seed" "conda activate layoutsim; python3 main.py --gpu $gpu --seed $train_zone$seed --multi_agent 1 --train_on $train_zone --prefix new_experiment_fix/ --std_decay_period 20"
        let gpu=(gpu+1)%3
    done
done
# tmux new-session -d -s "bash-session-" "conda activate layoutsim; python3 main.py --gpu $gpu --seed $seed --multi_agent 1 --train_on $train_zone --prefix new_experiment/"