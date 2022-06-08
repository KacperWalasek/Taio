"""
Script for creating sbatch files for SLURM.
"""

import os

with open("start_template", "r", encoding='UTF-8') as file:
    template = file.read()

for series_data_dir in os.scandir("tests"):
    if series_data_dir.is_dir():
        class_count = sum(
            1 for x in os.scandir(next(os.scandir(series_data_dir)).path) if x.is_dir()
        )
        if class_count > 6:
            CPUS = 10
        elif class_count > 4:
            CPUS = 4
        elif class_count > 2:
            CPUS = 2
        else:
            CPUS = 1
        formatted_template = template.format(
            cpus=CPUS, name=series_data_dir.name
        )
        with open(f"starts/start_{series_data_dir.name}", "w", encoding='UTF-8') as file:
            file.write(formatted_template)
