import os
import json
import argparse
from pathlib import Path

def combine_json_lists(input_dir: Path, output_file: Path):
    """
    Combines a folder of JSON files containing lists into a single JSON file.

    The script merges the lists by replacing null values. It ensures that there are
    no conflicts where the same index is non-null in multiple files.

    Args:
        input_dir: The directory containing the JSON files to merge.
        output_file: The path to the output JSON file.
    """
    json_files = sorted(input_dir.glob('*.json'))
    if not json_files:
        print(f"No JSON files found in '{input_dir}'.")
        return

    print(f"Found {len(json_files)} JSON files to combine.")

    merged_data = None
    first_file_path = None

    for json_file in json_files:
        with open(json_file, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from '{json_file}'. Skipping.")
                continue

        if not isinstance(data, list):
            print(f"Warning: Expected a list in '{json_file}', but found {type(data)}. Skipping.")
            continue

        if merged_data is None:
            merged_data = data
            first_file_path = json_file
            print(f"Initialized merged data with '{json_file}' (length: {len(merged_data)}).")
            continue

        if len(data) != len(merged_data):
            print(f"Warning: Length of list in '{json_file}' ({len(data)}) does not match "
                  f"the length of the first file '{first_file_path}' ({len(merged_data)}). Skipping.")
            continue

        for i, item in enumerate(data):
            if item is not None:
                if merged_data[i] is not None:
                    raise ValueError(
                        f"Conflict found in '{json_file}' at index {i}. "
                        f"The index is already populated by a previous file."
                    )
                merged_data[i] = item
    
    if merged_data is not None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(merged_data, f, indent=2)
        print(f"Successfully combined JSON files into '{output_file}'.")
    else:
        print("No valid JSON files were found to combine.")


def main():
    parser = argparse.ArgumentParser(
        description="Combine multiple JSON lists from a folder into a single list, "
                    "handling null value replacement and conflict detection.",
        epilog="Example: python tools/combine_json_lists.py keypoints/activity1 merged_keypoints.json"
    )
    parser.add_argument('input_dir', type=str, help='Path to the directory containing the JSON files.')
    parser.add_argument('output_file', type=str, help='Path to the output combined JSON file.')
    
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_file)

    if not input_path.is_dir():
        print(f"Error: Input directory '{input_path}' not found.")
        return

    try:
        combine_json_lists(input_path, output_path)
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
