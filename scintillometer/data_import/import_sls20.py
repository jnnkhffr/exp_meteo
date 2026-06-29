""" This Skript reads in the data from the SLS20 Scintillometer (small one)."""

from pathlib import Path

folder = Path("C:/Users/janni/Desktop/Studium/Master/Experimental Meteo/sample_data/SLS20")

for file in folder.iterdir():
    # Skip hidden/system files
    if file.name.startswith("."):
        print(f"Skipping hidden file: {file.name}")
        continue

    print(f"Reading file: {file.name}")
    with open(file, "r", errors="ignore") as f:
        content = f.read()
        print(content[:100])  # preview
