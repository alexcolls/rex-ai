import os
import time
import pandas as pd
from datetime import datetime, timedelta, timezone
from google.cloud import bigquery
from gbq_utils import get_table_last_date

PROJECT = "artful-talon-355716"
DATASET = "rex_ai"
LOCATION = "EU"

dataset_id = f"{PROJECT}.{DATASET}"
client = bigquery.Client()


def upload_tendency_volatility_data():
    DATA_PATH = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../", "data", "merge")
    )
    DIRS = ["tendency", "volatility"]

    try:
        client.create_dataset(bigquery.Dataset(dataset_id), timeout=30)
        print(f"Created empty dataset {dataset_id}")
    except Exception:
        print(f"Dataset {dataset_id} exists")

    for d in DIRS:
        path = os.path.join(DATA_PATH, d)

        last_gbq_date = get_table_last_date(client, dataset_id, d)

        for file in os.listdir(path):

            start_time = time.time()

            data = pd.read_csv(os.path.join(path, file), index_col=0)
            data.index = pd.to_datetime(data.index)
            data.index.name = "DATE_TIME"
            FIRST_VALID_HOUR = last_gbq_date or datetime(
                2010, 1, 1, 0, 0, 0, 0, timezone.utc
            )
            LAST_VALID_HOUR = datetime.now(timezone.utc)
            date_diff = LAST_VALID_HOUR - FIRST_VALID_HOUR
            hour_diff = date_diff.seconds // 3600

            data = data.loc[FIRST_VALID_HOUR + timedelta(0, 3600) : LAST_VALID_HOUR]

            if hour_diff > 0 and len(data) > 0:
                print(f"Uploading last {hour_diff} hours from {file} to {dataset_id}")

                try:
                    job_config = bigquery.LoadJobConfig(
                        write_disposition="WRITE_APPEND"
                    )
                    job = client.load_table_from_dataframe(
                        data,
                        f"{dataset_id}.{d}",
                        job_config=job_config,
                    )  # Make an API request.
                    job.result()
                except Exception as e:
                    print(e)

                print(f"Uploaded successfully in {round(time.time() - start_time, 2)}s")
            else:
                print(f"{dataset_id}.{d} up to date")

    return


def upload_csv_data(parent: str, files: list, year: int = 2022):
    DATA_PATH = os.path.normpath(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../",
            "data",
            parent,
            str(year),
        )
    )

    FILES = files

    try:
        client.create_dataset(bigquery.Dataset(dataset_id), timeout=30)
        print(f"Created empty dataset {dataset_id}")
    except Exception:
        print(f"Dataset {dataset_id} exists")

    for f in FILES:

        file_name = f"{f}.csv"
        last_gbq_date = get_table_last_date(client, dataset_id, f)
        start_time = time.time()

        path = os.path.join(DATA_PATH, file_name)

        try:
            data = pd.read_csv(path, index_col=0)
        except FileNotFoundError:
            print(f"File {file_name} not found")
            return

        data.index = pd.to_datetime(data.index)
        data.index.name = "DATE_TIME"
        FIRST_VALID_HOUR = last_gbq_date or datetime(
            2010, 1, 1, 0, 0, 0, 0, timezone.utc
        )
        LAST_VALID_HOUR = datetime.now(timezone.utc)
        date_diff = LAST_VALID_HOUR - FIRST_VALID_HOUR
        hour_diff = date_diff.seconds // 3600

        data = data.loc[FIRST_VALID_HOUR + timedelta(0, 3600) : LAST_VALID_HOUR]

        if hour_diff > 0 and len(data) > 0:
            print(f"Uploading last {hour_diff} hours from {file_name} to {dataset_id}")

            try:
                job_config = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")
                job = client.load_table_from_dataframe(
                    data,
                    f"{dataset_id}.{f}",
                    job_config=job_config,
                )  # Make an API request.
                job.result()
            except Exception as e:
                print(e)

            print(f"Uploaded successfully in {round(time.time() - start_time, 2)}s")
        else:
            print(f"{dataset_id}.{f} up to date")

    return


def upload_dataframe(df: pd.DataFrame, name: str):

    try:
        client.create_dataset(bigquery.Dataset(dataset_id), timeout=30)
        print(f"Created empty dataset {dataset_id}")
    except Exception:
        print(f"Dataset {dataset_id} exists")

    last_gbq_date = get_table_last_date(client, dataset_id, name)
    start_time = time.time()

    df.index = pd.to_datetime(df.index)
    df.index.name = "DATE_TIME"
    FIRST_VALID_HOUR = last_gbq_date or datetime(2010, 1, 1, 0, 0, 0, 0, timezone.utc)
    df = df.loc[FIRST_VALID_HOUR + timedelta(0, 3600) :]

    if len(df) > 0:
        print(f"Uploading last {len(df)} hours from {name} to {dataset_id}")

        try:
            job_config = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")
            job = client.load_table_from_dataframe(
                df,
                f"{dataset_id}.{name}",
                job_config=job_config,
            )  # Make an API request.
            job.result()
        except Exception as e:
            print(e)

        print(f"Uploaded successfully in {round(time.time() - start_time, 2)}s")
    else:
        print(f"{dataset_id}.{name} up to date")

    return


if __name__ == "__main__":
    upload_csv_data("primary", ["closes"])
    upload_csv_data("tertiary", ["logs_"])
    upload_tendency_volatility_data()
