"""
To nie bedzie wysylane
"""
import os
import sys
import read_data


def remove_short(path, limit):  # pylint: disable=missing-function-docstring
    class_dirs = [
        (entry.name, entry.path) for entry in os.scandir(path) if entry.is_dir()
    ]

    removed = 0
    for class_dir in class_dirs:
        series_list = []
        for file in os.scandir(class_dir[1]):
            if file.name.endswith(".csv"):
                series = read_data.process_data(file.path, 1)
                if len(series[0]) < limit:
                    print("remove", file.path)
                    removed += 1
                    os.remove(file.path)
                else:
                    series_list.append(series)
    print("removed", removed, "files")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise RuntimeError(
            "Too few arguments, dataset directory path or limit is missing"
        )
    data_dir = sys.argv[1]
    min_limit = int(sys.argv[2]) + 1

    train_path = os.path.join(data_dir, "Train")
    test_path = os.path.join(data_dir, "Test")
    print("Train")
    remove_short(train_path, min_limit)
    print()
    print("Test")
    remove_short(test_path, min_limit)
