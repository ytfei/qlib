FROM python:3.13-slim

WORKDIR /qlib

COPY . .

RUN apt-get update && \
    apt-get install -y build-essential

RUN python -m pip install --no-cache-dir --upgrade pip && python -m pip install --no-cache-dir -e .
