#!/bin/bash

#SBATCH --account=<REMOVED FOR PRIVACY>
#SBATCH --array=0-3
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

python3 6-transfer_policies.py --seed ${seed_prefix}${SLURM_ARRAY_TASK_ID} --episodes 5000 --continue_train 0 --from_scratch ${scratch} --clusters ${clusters} --continue_train 0 --no_blinds ${noblind}