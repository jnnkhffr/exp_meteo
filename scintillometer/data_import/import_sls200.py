""" This Skript reads in the data from the SLS20 Scintillometer (small one)."""

from pathlib import Path
import pandas as pd

# folder = Path("C:/Users/janni/Desktop/Studium/Master/Experimental Meteo/sample_data/SLS20")

# for file in folder.iterdir():
# Skip hidden/system files
#    if file.name.startswith("."):
#        print(f"Skipping hidden file: {file.name}")
#        continue

#    print(f"Reading file: {file.name}")
#    with open(file, "r", errors="ignore") as f:
#        content = f.read()
#        print(content[:100])  # preview


def load_sls20_data(folder_path: str):
    folder = Path(folder_path)

    dgn_data = {}
    res_data = {}

    for file in folder.iterdir():

        # Skip hidden/system files
        if file.name.startswith("."):
            print(f"Skipping hidden file: {file.name}")
            continue

        # DGN files
        if file.suffix.lower() == ".dgn":
            print(f"Parsing DGN file: {file.name}")
            with open(file, "r", errors="ignore") as f:
                lines = f.readlines()

            # Convert each line into a list of floats
            parsed_lines = []
            for line in lines:
                parts = line.split()
                parts = [float(p) for p in parts]
                parsed_lines.append(parts)

            dgn_data[file.stem] = parsed_lines

        # RES files
        elif file.suffix.lower() == ".res":
            print(f"Parsing RES file: {file.name}")

            # RES files are whitespace-separated
            df = pd.read_csv(file, sep=r"\s+", header=None)
            res_data[file.stem] = df

        # Other files (e.g., SLSLOG.DAT)
        else:
            print(f"Skipping non-data file: {file.name}")

    return dgn_data, res_data


if __name__ == "__main__":
    folder = (
        "C:/Users/janni/Desktop/Studium/Master/Experimental Meteo/sample_data/SLS20"
    )
    dgn, res = load_sls20_data(folder)

    print("\nLoaded DGN files:", list(dgn.keys()))
    print("Loaded RES files:", list(res.keys()))

print("stop")
