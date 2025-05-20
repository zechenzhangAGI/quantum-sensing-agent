import sys
import json
import os
import importlib.util
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
sys.path.append(r'C:\Users\NVAFM_6th_fl_2\NV-Automation\b26_toolkit_for_agent\b26_toolkit-master')
from pylabcontrol.core import Script
from b26_toolkit.scripts.galvo_scan.galvo_scan import GalvoScan
def main():
    """
    Usage:
<<<<<<< HEAD
      py galvo_scan.py --config configs/galvo_experiment_YYYY-MM-DD.json
=======
      py galvo_scan.py --config configs/gavlo_experiment_YYYY-MM-DD.json [--output-dir path/to/output/directory]
>>>>>>> origin/master
    """
    # 1. Parse command-line args
    parser = argparse.ArgumentParser(description='Run GalvoScan experiment')
    parser.add_argument('--config', required=True, help='Path to the config JSON file')
    parser.add_argument('--output-dir', default='data', help='Directory to save output data and plots')
    args = parser.parse_args()
    
    config_file = args.config
    data_dir = args.output_dir
    if not os.path.exists(config_file):
        print(f"[Runner] Config file not found: {config_file}")
        sys.exit(1)

    # 2. Load the JSON config
    with open(config_file, "r") as f:
        config_data = json.load(f)

    # The relevant ESR section typically lives at config_data["scripts"]["esr_RnS"]
    info = config_data["scripts"]["galvo_scan"]
    script_path = info["filepath"]  
  

    if not os.path.exists(script_path):
        print(f"[Runner] ESR script not found at: {script_path}")
        sys.exit(1)

    # 4. Instantiate the GalvoScan class
    GS = GalvoScan(config_file=config_file)
    print("[Runner] Created GalvoScan instance.")

    # 5. Run the actual ESR measurement by calling the `_function()` method
    #    (This is the method in your ESR_RnS code that does the measurement.)
    print("[Runner] Starting GalvoScan ...")
    GS._function()  # This will run the entire ESR sequence
    print("[Runner] Galvo Scan measurement completed!")

    # 6. Plot the results. ESR_RnS._plot() expects a list of axes
    #    We'll create a single figure + single axis, then pass it as [axis].
    fig, ax = plt.subplots(figsize=(4,3))
    # The ESR code's `_plot()` checks for `axes_list[0]`—so passing [ax] is enough.
    GS._plot([ax], data=GS.data)  # Plot the final ESR data

    # 7. Save the figure to the specified output directory with a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# <<<<<<< HEAD
#     data_dir = "data"  # Adjust if your layout differs -- PASS IN A COMMAND LINE ARGUMENT HERE
# =======
# >>>>>>> origin/master
    os.makedirs(data_dir, exist_ok=True)
    outpath = os.path.join(data_dir, f"GalvoScan_plot_{timestamp}.png")
    fig.savefig(outpath, dpi=150)
    print(f"[Runner] Saved GalvoScan plot to: {outpath}")

    #8. Optionally, also save esr.data as a JSON or pickle if you wish
    #   e.g.:
    outjson = os.path.join(data_dir, f"GalvoScan_data_{timestamp}.json")
    with open(outjson, "w") as f:
        json.dump(GS.data, f, indent=2, default=str)  # default=str for numpy conversions
    print(f"[Runner] Saved GalvoScan data to: {outjson}")

if __name__ == "__main__":
    main()