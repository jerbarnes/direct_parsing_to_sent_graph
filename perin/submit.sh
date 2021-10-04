#!/bin/bash

id=$(sbatch run.sh)
watch "cat slurm-${id: -7}.out | tail -n 37"