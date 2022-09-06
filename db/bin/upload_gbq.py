import os
import time
import pandas as pd
from datetime import datetime
from pytz import UTC
from google.cloud import bigquery

PROJECT = "artful-talon-355716"
DATASET = "rex_ai"
LOCATION = "EU"

dataset_id = f"{PROJECT}.{DATASET}"

client = bigquery.Client()
dataset = bigquery.Dataset(dataset_id)
dataset.location = LOCATION


def upload_data():
    DATA_PATH = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../", "data", "merge")
    )
    DIRS = ["tendency", "volatility"]

    try:
        client.create_dataset(bigquery.Dataset(dataset_id), timeout=30)
        print(f"Created dataset {dataset_id}", flush=True, end="\r")
    except Exception:
        print(f"Dataset {dataset_id} already exists", flush=True, end="\r")

    for d in DIRS:
        path = os.path.join(DATA_PATH, d)

        # Check last data available on GBQ

        for file in os.listdir(path):

            start_time = time.time()
            print(f"Uploading {file} to {dataset_id}", flush=True, end="\r")
            data = pd.read_csv(os.path.join(path, file), index_col=0)
            data.index = pd.to_datetime(data.index)
            data.index.name = "DATE_TIME"
            FIRST_VALID_DAY = datetime(2010, 1, 1, 0, 0, 0, 0, UTC)
            data = data.loc[FIRST_VALID_DAY:]

            try:
                job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
                job = client.load_table_from_dataframe(
                    data, f"{dataset_id}.{file.split('.')[0]}", job_config=job_config
                )  # Make an API request.
                job.result()
            except Exception as e:
                print(e)

            print(f"Uploaded {file} in {round(time.time() - start_time, 2)}s\t\t")

    return


if __name__ == "__main__":
    upload_data()
