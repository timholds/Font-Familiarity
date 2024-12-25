import wandb
import subprocess
import time
import sys

def run_sweep(sweep_id):
    # Get the sweep configuration
    api = wandb.Api()
    sweep = api.sweep(sweep_id)
    
    # Calculate total number of runs from parameter space
    params = sweep.config['parameters']
    num_runs = 1
    for param, config in params.items():
        if 'values' in config:
            num_runs *= len(config['values'])
    
    print(f"Starting sweep with {num_runs} total runs")
    
    # Run the sweeps sequentially
    for i in range(num_runs):
        print(f"Starting run {i+1} of {num_runs}")
        subprocess.run(['wandb', 'agent', sweep_id, '--count', '1'])
        time.sleep(5)  # Brief pause between runs

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_sweep.py SWEEP_ID")
        sys.exit(1)
    
    sweep_id = sys.argv[1]
    run_sweep(sweep_id)