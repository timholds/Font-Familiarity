#!/bin/bash
SWEEP_ID=$1

# Hardcode the number of runs (3*2*2*2*2 = 48) or just run a large number
NUM_RUNS=48

echo "Starting sweep with up to $NUM_RUNS runs"
for i in $(seq 1 $NUM_RUNS); do
    echo "Starting run $i of $NUM_RUNS"
    wandb agent $SWEEP_ID --count 1
    sleep 5
done