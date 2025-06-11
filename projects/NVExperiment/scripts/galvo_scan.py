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
from b26_toolkit.scripts.galvo_scan.galvo_scan import GalvoScan

def numpy_to_python(obj):
    """Convert nested dictionary with numpy arrays to Python native types."""
    if isinstance(obj, dict):
        return {k: numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_python(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, datetime):
        return obj.strftime("%Y-%m-%d %H:%M:%S")
    return obj

def main():
    """
    Usage:
      py galvo_scan.py --config configs/galvo_scan_experiment_YYYY-MM-DD.json [--output-dir path/to/output/directory]
    """
    # Parse command-line args
    parser = argparse.ArgumentParser(description='Run GalvoScan experiment')
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

    # The relevant GalvoScan section typically lives at config_data["scripts"]["galvo_scan"]
    galvo_scan_info = config_data["scripts"]["galvo_scan"]
    script_path = galvo_scan_info["filepath"]

    if not os.path.exists(script_path):
        print(f"[Runner] GalvoScan script not found at: {script_path}")
        sys.exit(1)

    # Instantiate the GalvoScan class
    galvo_scan = GalvoScan(config_file=config_file)
    print("[Runner] Created GalvoScan instance.")

    # Run the actual GalvoScan measurement
    print("[Runner] Starting GalvoScan measurement...")
    galvo_scan._function()
    print("[Runner] GalvoScan measurement completed!")

    # Plot the results
    fig, ax = plt.subplots(figsize=(6,4))
    galvo_scan._plot([ax], data=galvo_scan.data)

    # Save the figure with a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(data_dir, exist_ok=True)
    outpath = os.path.join(data_dir, f"galvo_scan_plot_{timestamp}.png")
    fig.savefig(outpath, dpi=150)
    print(f"[Runner] Saved GalvoScan plot to: {outpath}")

    # Save galvo_scan.data as a JSON with proper numpy array handling
    outjson = os.path.join(data_dir, f"galvo_scan_data_{timestamp}.json")
    with open(outjson, "w") as f:
        json.dump(galvo_scan.data, f, indent=4)
    print(f"[Runner] Saved GalvoScan data to: {outjson}")


if __name__ == "__main__":
    main()