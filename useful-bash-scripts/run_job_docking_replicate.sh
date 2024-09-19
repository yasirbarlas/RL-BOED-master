#!/bin/bash
for id in {1..10}
do
   sbatch run_job_docking_id.sh $id
done