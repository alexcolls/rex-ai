import os
import time
import pandas as pd
from datetime import datetime, timezone
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
        print(f"Created empty dataset {dataset_id}")
    except Exception:
        print(f"Dataset {dataset_id} exists")

    for d in DIRS:
        path = os.path.join(DATA_PATH, d)

        # Check last data available on GBQ
        query = f"""
            SELECT DATE_TIME
            FROM `{dataset_id}.{d}`
            ORDER BY DATE_TIME DESC
            LIMIT 1
        """

        last_gbq_date = None

        try:
            query_job = client.query(query)
            for row in query_job:
                last_gbq_date = row[0]
                break
        except Exception:
            print(f"No data found in {dataset_id}.{d}")

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

            if hour_diff > 0:
                print(f"Uploading last {hour_diff} hours from {file} to {dataset_id}")
                data = data.loc[FIRST_VALID_HOUR:LAST_VALID_HOUR]

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


if __name__ == "__main__":
    upload_data()
