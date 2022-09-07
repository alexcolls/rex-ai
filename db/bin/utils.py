import datetime
from typing import Union
from datetime import datetime


def get_table_last_date(client, dataset: str, table: str) -> Union[datetime, None]:

    table_fqn = f"{dataset}.{table}"

    # Check last data available on GBQ
    query = f"""
        SELECT DATE_TIME
        FROM `{table_fqn}`
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
        print(f"No data found in {table_fqn}")

    return last_gbq_date
