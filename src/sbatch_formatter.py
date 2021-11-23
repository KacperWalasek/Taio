import os

with open("start_template", "r") as file:
    template = file.read()

for series_data_dir in os.scandir("tests"):
    if series_data_dir.is_dir():
        class_count = sum(
            1 for x in os.scandir(next(os.scandir(series_data_dir)).path) if x.is_dir()
        )
        formatted_template = template.format(
            cpus=min(class_count * (class_count - 1), 128), name=series_data_dir.name
        )
        with open(f"start_{series_data_dir.name}", "w") as file:
            file.write(formatted_template)
