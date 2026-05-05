#!/bin/bash
#SBATCH --job-name=SAM_TRAIN
#SBATCH --partition=q15d              # node005 belongs to this partition
#SBATCH --nodelist=node005            # Direct hit on the node with 28 free CPUs
#SBATCH --gres=gpu:1                  # Requesting 1 of the 8 available GPUs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16            # Use 16 of the 28 free CPUs for fast patching
#SBATCH --mem=128G                    # High RAM for image processing
#SBATCH --time=2-00:00:00             # Set to 2 days (format: D-HH:MM:SS) 
#SBATCH --output=SAM_%j.out
#SBATCH --error=SAM_%j.err

# --- 1. Environment Setup ---
module purge
module load cuda/12.8 || module load cuda/12.1 # Fallback if 12.8 is missing

# --- 2. Activate Conda ---
source /home/cs24m119/miniconda3/etc/profile.d/conda.sh
conda activate base

# --- 3. Sanity Checks ---
if [ -z "$1" ]; then
    echo "Error: No python script provided."
    echo "Usage: sbatch submit_gpu.sh path/to/script.py"
    exit 1
fi

echo "------------------------------------------------"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on Node: $SLURM_NODELIST"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "GPU assigned: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "------------------------------------------------"

# --- 4. Run the Training/Patching ---
# We use 'python -u' to ensure logs write to the .out file in real-time
python -u "$1"