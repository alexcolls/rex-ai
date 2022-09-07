import pandas as pd
from datetime import datetime, timedelta, timezone
from google.cloud import bigquery
from utils import get_table_last_date

PROJECT = "artful-talon-355716"
DATASET = "rex_ai"
LOCATION = "EU"

dataset_id = f"{PROJECT}.{DATASET}"
client = bigquery.Client()


def load_last_rows(table: str, row_number: int = 120):

    table_fqn = f"{dataset_id}.{table}"

    # Check last data available on GBQ
    query = f"""
        SELECT *
        FROM `{table_fqn}`
        ORDER BY DATE_TIME DESC
        LIMIT {row_number}
    """

    try:
        df = client.query(query).to_dataframe()
        df.set_index("DATE_TIME", inplace=True)
        df.index = pd.to_datetime(df.index)
    except Exception:
        print(f"No data found in {table_fqn}")
        df = None

    return df


if __name__ == "__main__":
    print(load_last_rows("tendency"))
