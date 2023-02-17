#!/bin/bash

#SBATCH --account=<REMOVED FOR PRIVACY>
#SBATCH --array=0-4
#SBATCH --time=120:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --mail-user=<REMOVED FOR PRIVACY>
#SBATCH --mail-type=ALL

# WARN: the order of the loaded modules matters
module load StdEnv/2020 energyplus/9.3.0
module load cuda cudnn
# load the virtual environment
source ~/env_py3.7/bin/activate

export OMP_NUM_THREADS=1

echo "prog started at: `date`"

python3 6-transfer_policies.py --seed 10${seed_prefix}${SLURM_ARRAY_TASK_ID} --episodes 5000 --continue_train 0 --single_agent ${sa} --flexlab ${fl} --ignore_zero ${iz} --from_scratch 1 --change_rotation_location 1 --doee 0