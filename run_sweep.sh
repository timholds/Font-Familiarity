#!/bin/bash
SWEEP_ID=$1

# Get the number of runs using wandb API
NUM_RUNS=$(wandb sweep $SWEEP_ID --pretty | grep "Total runs needed:" | awk '{print $4}')

echo "Starting sweep with $NUM_RUNS total runs"
for i in $(seq 1 $NUM_RUNS); do
    echo "Starting run $i of $NUM_RUNS"
    wandb agent $SWEEP_ID --count 1
    sleep 5
done