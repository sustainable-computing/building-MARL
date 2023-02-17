#!/bin/bash

#SBATCH --account=<REMOVED FOR PRIVACY>
#SBATCH --array=0-10
#SBATCH --time=40:00:00
#SBATCH --cpus-per-task=2
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

python3 4-evaluate_library_on_doee.py --evaluate_on ${SLURM_ARRAY_TASK_ID} --evaluate_start ${start} --evaluate_length 50