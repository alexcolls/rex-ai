# Docker build and run instructions

## Build

```sh
docker build -t rex-ai-data docker/
```

## Create volume

```sh
docker volume create rex-ai-data
```

## Run container

Using volume `rex-ai-data` mapped to `/data`.

```sh
 docker run -d -v rex-ai-data:/data rex-ai-data:latest
```
