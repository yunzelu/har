import subprocess
import argparse
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# --- CONFIGURATION ---
DEFAULT_DATA_DIR = r"D:\lu\project\har\data\up_dataset\Walking"
MIRROR_SCRIPT_PATH = "./tools/dataset/mirror_json.py"
NUM_WORKERS = 5

# --- FUNCTIONS ---
def parse_arguments():
    """Parses command line arguments for the batch job."""
    parser = argparse.ArgumentParser(description="Batch process JSON files for mirroring.")
    parser.add_argument("--dir", type=str, default=DEFAULT_DATA_DIR, 
                        help="Root directory containing the JSON files.")
    return parser.parse_args()

def run_mirror_job(input_json_path, script_path):
    """Constructs the output filename and calls the mirror script as a subprocess."""
    input_path = Path(input_json_path)
    
    # Generate the new name: original_mirrored.json
    output_name = f"{input_path.stem}_mirrored.json"
    output_path = input_path.parent / output_name
    
    # Build the command to call the other script
    command = [
        sys.executable,  # Uses your current virtual environment's Python
        script_path,
        "--input", str(input_path),
        "--output", str(output_path)
    ]
    
    # Run the script silently
    result = subprocess.run(command, capture_output=True, text=True)
    
    if result.returncode != 0:
        return f"Error processing {input_path.name}: {result.stderr}"
    return f"Successfully mirrored {input_path.name}"

def batch_process(data_dir, script_path, max_workers):
    """Finds all target JSONs and processes them in parallel."""
    data_path = Path(data_dir)
    
    # Find all JSONs but ignore ones that are already mirrored
    all_jsons = list(data_path.rglob("*.json"))
    target_jsons = [f for f in all_jsons if not f.name.endswith("_mirrored.json")]
    
    print(f"Found {len(target_jsons)} files to process in {data_dir}")
    
    # Spin up the parallel workers
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs to the pool
        futures = {
            executor.submit(run_mirror_job, json_file, script_path): json_file 
            for json_file in target_jsons
        }
        
        # Use tqdm to show a progress bar as the jobs complete
        for future in tqdm(as_completed(futures), total=len(futures), desc="Mirroring Files"):
            try:
                # You can print future.result() here if you want to see the success/error messages
                future.result() 
            except Exception as e:
                print(f"\nJob crashed: {e}")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    args = parse_arguments()
    
    # Check if the mirroring script actually exists before starting
    if not Path(MIRROR_SCRIPT_PATH).exists():
        print(f"Error: Could not find '{MIRROR_SCRIPT_PATH}' in the current folder.")
        sys.exit(1)
        
    batch_process(args.dir, MIRROR_SCRIPT_PATH, NUM_WORKERS)