import json
from pathlib import Path
import pandas as pd
from collections import defaultdict
import re
import argparse

# Define all libraries to process
LIBRARIES = ["matplotlib", "seaborn", "plotly"]

def calculate_success_rates(data):
    # Dictionary to store results for all libraries
    all_results = defaultdict(lambda: defaultdict(dict))
    
    for key, value in data.items():
        if isinstance(value, float):
            continue
            
        ckpt_path, lib = key.rsplit("_", 1)
        if lib not in LIBRARIES:
            continue
            
        model_name = Path(ckpt_path).name
        total_cases = value["total_num"]
        
        # Calculate initial success rate
        init_success = ((total_cases - value["execution_error_num"]) / total_cases) * 100
        all_results[lib][model_name]["Init"] = init_success
        
        # Calculate success rates for each attempt
        if "debug_attempts" in value:
            for attempt in range(3):  # A0, A1, A2
                attempt_key = f"attempt_{attempt}"
                if attempt_key in value["debug_attempts"]:
                    attempt_stats = value["debug_attempts"][attempt_key]
                    success_rate = ((total_cases - attempt_stats["execution_error_num"]) / total_cases) * 100
                    all_results[lib][model_name][f"Post A{attempt}"] = success_rate
                else:
                    all_results[lib][model_name][f"Post A{attempt}"] = None
    
    return all_results

def create_formatted_table(results, lib):
    df = pd.DataFrame.from_dict(results[lib], orient="index")
    
    df.index.name = "Model"
    df.reset_index(inplace=True)
    
    # Ensure all expected columns are present
    expected_columns = ["Model", "Init", "Post A0", "Post A1", "Post A2"]
    for col in expected_columns:
        if col not in df.columns:
            df[col] = None  # Add missing columns with None values
    
    # Round values
    for col in df.columns[1:]:
        df[col] = df[col].round(2)
    
    return df

def main():
    parser = argparse.ArgumentParser(description="Print execution success rates from JSON results.")
    parser.add_argument("input_file", type=str, help="Path to the input JSON file.")
    parser.add_argument("--libs", type=str, nargs="*", default=LIBRARIES, help="Specific libraries to print results for (default: all).")
    args = parser.parse_args()

    # Load data
    with open(args.input_file, "r") as f:
        data = json.load(f)
    
    # Calculate success rates for all libraries
    all_results = calculate_success_rates(data)
    
    # Print results for each specified library
    for lib in args.libs:
        if lib in all_results:
            print(f"\n=== {lib.upper()} Execution Success Rates (%) ===")
            df = create_formatted_table(all_results, lib)
            print(df.to_markdown(index=False))
            print("\n")
        else:
            print(f"\nNo data available for library: {lib}\n")

if __name__ == "__main__":
    main()
