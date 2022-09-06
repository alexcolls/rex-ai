import os
import pandas as pd
from pathlib import Path


def merge_db_data():
    DATA_PATH = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../", "data")
    )

    dirs = ["primary", "secondary", "tertiary"]

    for d in dirs:
        print(f"Process {d} data:")
        path = os.path.join(DATA_PATH, d)
        if len(os.listdir(path)) != 0:
            print(f"Directory {path} is not empty. Skipping.")
            continue
        merge = {}
        for year_dir in sorted(os.listdir(path)):
            if "." in year_dir:
                continue
            year_path = os.path.join(path, year_dir)
            for file in os.listdir(year_path):
                file_path = os.path.join(year_path, file)
                file_name = file.split(".")[0]
                data = pd.read_csv(file_path, index_col=0)
                merge[file_name] = pd.concat(
                    [merge.get(file_name, pd.DataFrame()), data]
                )
                print(f"Processing {year_dir} - {file}", flush=True, end="\r")

        for k in merge:
            file_path = os.path.join(DATA_PATH, "merge", d)
            Path(file_path).mkdir(parents=True, exist_ok=True)
            merge[k].to_csv(os.path.join(file_path, f"{k}.csv"))
            print(f"{k} shape {merge[k].shape}    ")


if __name__ == "__main__":
    merge_db_data()
