#!/bin/bash

#SBATCH --account=<REMOVED FOR PRIVACY>
#SBATCH --array=0-14
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --mail-user=<REMOVED FOR PRIVACY>
#SBATCH --mail-type=ALL

# WARN: the order of the loaded modules matters
module load StdEnv/2020 energyplus/9.3.0
module load cuda cudnn
# load the virtual environment
source ~/env_py3.7/bin/activate

export OMP_NUM_THREADS=1

echo "prog started at: `date`"

python3 3-sample_actions_for_history_states.py --evaluate_on ${SLURM_ARRAY_TASK_ID} --change_rotation_location 1