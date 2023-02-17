#!/bin/bash

#SBATCH --account=<REMOVED FOR PRIVACY>
#SBATCH --array=0-9
#SBATCH --time=200:00:00
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
python3 1-5zone_training.py --seed ${start_month}${ignore}${SLURM_ARRAY_TASK_ID} --start_month ${start_month} --prefix 5_zone_ignore_${ignore}_blind_${blind}_OPTIMAL_ --blind ${blind} --ignore_zero_reward ${ignore} --episodes 5000