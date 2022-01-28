#!/bin/bash

tmux new-session -d -s "bash-session-$2$3" "conda activate layoutsim; python3 main.py --gpu $1 --seed $2$3 --multi_agent 1 --train_on $2 --prefix new_experiment/ --std_decay_period 20"

# tmux new-session -d -s "bash-session-" "conda activate layoutsim; python3 main.py --gpu $gpu --seed $seed --multi_agent 1 --train_on $train_zone --prefix new_experiment/"