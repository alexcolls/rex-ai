FROM python:3.9-slim

COPY rex-ai-data /docker
COPY requirements.txt docker/requirements.txt

RUN pip install --upgrade pip
RUN pip install -r docker/requirements.txt

CMD mkdir -p /data/primary; mkdir -p /data/secondary; mkdir -p /data/tertiary; python /docker/update_db.py
