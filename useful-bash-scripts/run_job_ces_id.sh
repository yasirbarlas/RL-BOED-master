#!/bin/bash
#SBATCH --job-name boed-ces                      # Job name
#SBATCH --partition=prigpu                       # Select the correct partition.
#SBATCH --nodelist=gpu04
#SBATCH --nodes=1                                # Run on 1 nodes (each node has 48 cores)
#SBATCH --ntasks-per-node=1                        # Run one task
#SBATCH --cpus-per-task=4                          # Use 4 cores, most of the procesing happens on the GPU
#SBATCH --mem=40GB                                 # Expected ammount CPU RAM needed (Not GPU Memory)
#SBATCH --time=72:00:00                            # Expected ammount of time to run Time limit hrs:min:sec
#SBATCH --gres=gpu:1                               # Use one gpu.
#SBATCH -e results/%x_%j.e                         # Standard output and error log [%j is replaced with the jobid]
#SBATCH -o results/%x_%j.o                         # [%x with the job name], make sure 'results' folder exists.
#SBATCH --error boed_err_ces.err
#SBATCH --output boed_out_ces.output

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
python3 Adaptive_CES_REDQ.py --n-parallel=100 --n-contr-samples=100000 --n-rl-itr=20001 --log-dir="boed_results_ensemble_10_ces/ces" --bound-type=lower --id=$1 --budget=10 --discount=0.9 --buffer-capacity=1000000 --M=2 --ens-size=10 --lstm-q-function=False --layer-norm=False --dropout=0


