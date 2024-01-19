#!/bin/bash
#SBATCH -A research
#SBATCH -n 39
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=3000M
#SBATCH --time=4-00:00:00
#SBATCH --job-name=all_experiments
#SBATCH --output=hyperparameter_search.out
#SBATCH --mail-user=tathagato.roy@research.iiit.ac.in
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
5




# Discord notifs on start and end
source notify

# Fail on error

#making the neccessary file structure
#cd /scratch

# Copy dataset
#rsync -azh --info=progress2 ada.iiit.ac.in:/share3/tathagato/hm3d-val-habitat /scratch/Habitat/

# Activate Conda environment
echo "Activating Conda Environment Virtual Environment"
source /home2/tathagato/miniconda3/bin/activate habitat

#scp -r gnode055:/scratch/tathagato /scratch/
./train_run.sh
#./test_model.sh
#./test_model_gpu_0.sh
#i./test_model_gpu_1.sh
#./run_script.sh



