import os
import glob

# Directory containing CSV files
input_dir = ""   # change this path if needed

# Iterate through all CSV files
for filepath in glob.glob(os.path.join(input_dir, "*.csv")):
    print(f"Processing {filepath}")

    # Read all lines except those containing "txt_lambda_0"
    with open(filepath, "r") as f:
        lines = f.readlines()

    filtered_lines = [
        line for line in lines
        if "txt_lambda_0" in line
    ]

    # Overwrite original file
    with open(filepath, "w") as f:
        f.writelines(filtered_lines)

print("All files processed.")
