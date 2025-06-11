# File: NV-automation/experiments/experiment_runner.py

import sys
import json
import os
import importlib.util
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
import numpy as np
sys.path.append(r'C:\Users\NVAFM_6th_fl_2\NV-Automation\b26_toolkit_for_agent\b26_toolkit-master')
from pylabcontrol.core import Script
from b26_toolkit.scripts.optimize import optimize

def numpy_to_python(obj):
    """Convert nested dictionary with numpy arrays to Python native types."""
    if isinstance(obj, dict):
        return {k: numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_python(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        # Handle 2D arrays specially to make them more readable
        if obj.ndim == 2:
            return [row.tolist() for row in obj]
        # For 1D arrays, if they're too long, summarize them
        elif obj.ndim == 1 and len(obj) > 10:
            return f"Array(length={len(obj)}, mean={obj.mean():.3f}, std={obj.std():.3f}, min={obj.min():.3f}, max={obj.max():.3f})"
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, datetime):
        return obj.strftime("%Y-%m-%d %H:%M:%S")
    return obj

class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that formats arrays nicely."""
    def __init__(self, *args, **kwargs):
        # Configure the encoder to use a reasonable number of spaces
        kwargs['indent'] = 2
        super().__init__(*args, **kwargs)
        
    def encode(self, obj):
        if isinstance(obj, str) and obj.startswith("Array("):
            # Don't add quotes around our special array summaries
            return obj
        return super().encode(obj)

def main():
    """
    Usage:
      py optimize.py --config configs/optimize_experiment_YYYY-MM-DD.json [--output-dir path/to/output/directory]
    """
    # Parse command-line args
    parser = argparse.ArgumentParser(description='Run Optimize experiment')
    parser.add_argument('--config', required=True, help='Path to the config JSON file')
    parser.add_argument('--output-dir', default='data', help='Directory to save output data and plots')
    args = parser.parse_args()
    
    config_file = args.config
    data_dir = args.output_dir
    if not os.path.exists(config_file):
        print(f"[Runner] Config file not found: {config_file}")
        sys.exit(1)

    # Load the JSON config
    with open(config_file, "r") as f:
        config_data = json.load(f)

    # The relevant Optimize section typically lives at config_data["scripts"]["optimize"]
    optimize_info = config_data["scripts"]["optimize"]
    script_path = optimize_info["filepath"]

    if not os.path.exists(script_path):
        print(f"[Runner] Optimize script not found at: {script_path}")
        sys.exit(1)

    # Instantiate the Optimize class
    optimize_instance = optimize(config_file=config_file)
    print("[Runner] Created Optimize instance.")

    # Run the actual Optimize measurement
    print("[Runner] Starting Optimize measurement...")
    optimize_instance._function()
    print("[Runner] Optimize measurement completed!")

    # Plot the results
    fig, ax = plt.subplots(figsize=(6,4))
    optimize_instance._plot([ax], data=optimize_instance.data)

    # Save the figure with a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(data_dir, exist_ok=True)
    outpath = os.path.join(data_dir, f"optimize_plot_{timestamp}.png")
    fig.savefig(outpath, dpi=150)
    print(f"[Runner] Saved Optimize plot to: {outpath}")

    # Save optimize.data as a JSON with proper numpy array handling
    outjson = os.path.join(data_dir, f"optimize_data_{timestamp}.json")
    with open(outjson, "w") as f:
        json.dump(optimize_instance.data, f, default=numpy_to_python)
    print(f"[Runner] Saved Optimize data to: {outjson}")

if __name__ == "__main__":
    main()