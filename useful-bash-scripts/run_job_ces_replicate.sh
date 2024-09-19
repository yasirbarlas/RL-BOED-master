#!/bin/bash
for id in {1..10}
do
   sbatch run_job_ces_id.sh $id
done