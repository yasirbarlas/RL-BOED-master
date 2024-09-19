#!/bin/bash
#SBATCH --job-name boed-s-eval                      # Job name
#SBATCH --partition=preemptgpu                       # Select the correct partition.
#SBATCH --nodelist=gpu01
#SBATCH --nodes=1                                # Run on 1 nodes (each node has 48 cores)
#SBATCH --ntasks-per-node=1                        # Run one task
#SBATCH --cpus-per-task=4                          # Use 4 cores, most of the procesing happens on the GPU
#SBATCH --mem=40GB                                 # Expected ammount CPU RAM needed (Not GPU Memory)
#SBATCH --time=72:00:00                            # Expected ammount of time to run Time limit hrs:min:sec
#SBATCH --gres=gpu:1                               # Use one gpu.
#SBATCH -e results/%x_%j.e                         # Standard output and error log [%j is replaced with the jobid]
#SBATCH -o results/%x_%j.o                         # [%x with the job name], make sure 'results' folder exists.
#SBATCH --error source_eval.err
#SBATCH --output source_eval.output

#Enable modules command

source /opt/flight/etc/setup.sh
flight env activate gridware
module load libs/nvidia-cuda/11.2.0/bin
module load gnu

python --version
#module load libs/nvidia-cuda/11.2.0/bin

#pip freeze
#Run your script.
export https_proxy=http://hpc-proxy00.city.ac.uk:3128
python3 select_policy_env.py --src="boed_results_source/source_9/itr_20000.pkl" --dest="boed_results_source/source_9/evaluation_lower.log" --edit_type="w" --seq_length=30 --bound_type=lower --n_contrastive_samples=1000000 --n_parallel=250 --n_samples=2000 --seed=1 --env="source" --source_d=2 --source_k=5 --source_b=0.1 --source_m=0.0001 --source_obs_sd=0.5 --ces_d=6 --ces_obs_sd=0.005 --docking_d=1

